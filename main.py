import os
import sys
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from uuid import UUID
from datetime import datetime
from dataclasses import dataclass, field

import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import Settings
from src.openai_client import OpenAIClient
from src.kg.builders.kg_builder import KGBuilder
from src.kg.manager.kg_manager import KGManager
from src.kg.storage.kg_repository import KGRepository
from src.api.agent_service import AgentService
from src.kg.storage.vector_store import VectorStore


LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "agent.log"


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (default: INFO)
        log_to_file: Write logs to file (default: True)
        log_to_console: Write logs to console/stderr (default: True)
    
    Returns:
        Root logger instance
    """
    # Create logs directory
    LOG_DIR.mkdir(exist_ok=True)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_to_file:
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Console handler (stderr to bypass Streamlit capture)
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Configure module loggers
    for module in ['src', 'src.agents', 'src.kg', 'src.api', 'src.orchestration', 'src.memory']:
        logging.getLogger(module).setLevel(level)
    
    # Suppress noisy third-party logs
    for lib in ['openai', 'httpx', 'urllib3', 'chromadb', 'langfuse']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    return root_logger

# Initialize logging
logger = setup_logging()

@dataclass
class ConnectionResult:
    """Result of database connection attempt"""
    success: bool
    source_conn: Optional[Any] = None
    kg_conn: Optional[Any] = None
    error: Optional[str] = None
    settings: Optional[Settings] = None

@dataclass
class KGBuildResult:
    """Result of Knowledge Graph build operation"""
    success: bool
    kg_id: Optional[UUID] = None
    tables_count: int = 0
    relationships_count: int = 0
    columns_count: int = 0
    message: str = ""
    error: Optional[str] = None
    kg_data: Optional[Dict[str, Any]] = None

@dataclass
class KGLoadResult:
    """Result of Knowledge Graph load operation"""
    success: bool
    kg_id: Optional[UUID] = None
    db_name: Optional[str] = None
    tables_count: int = 0
    relationships_count: int = 0
    columns_count: int = 0
    error: Optional[str] = None
    kg_data: Optional[Dict[str, Any]] = None

@dataclass
class KGListItem:
    """Knowledge Graph list item"""
    kg_id: UUID
    db_name: str
    db_host: str
    db_port: int
    status: str
    tables_count: int
    created_at: datetime
    last_updated: datetime

@dataclass
class QueryResult:
    """Result of query processing"""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    sql: Optional[str] = None
    explanation: Optional[str] = None
    error: Optional[str] = None
    error_category: Optional[str] = None
    needs_clarification: bool = False
    clarification_request: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None

@dataclass
class FeedbackResult:
    """Result of feedback submission"""
    success: bool
    lesson_extracted: bool = False
    error: Optional[str] = None

@dataclass
class ProgressUpdate:
    """Progress update for UI feedback"""
    stage: str           # e.g., "schema_extraction", "sql_generation", "execution"
    message: str         # Human-readable message
    progress: float      # 0.0 to 1.0
    details: Optional[Dict[str, Any]] = None


# Type alias for progress callback
ProgressCallback = Callable[[ProgressUpdate], None]


def _default_progress_callback(update: ProgressUpdate) -> None:
    """Default progress callback that logs to logger"""
    logger.info(f"[{update.stage}] {update.message} ({update.progress*100:.0f}%)")


def get_kg_connection() -> ConnectionResult:
    """
    Create KG storage database connection only (from settings/environment).
    
    Returns:
        ConnectionResult with kg_conn or error
    """
    logger.info("Creating KG storage connection...")
    
    try:
        settings = Settings()
        
        # KG storage connection (from environment/settings)
        kg_conn = psycopg2.connect(
            host=settings.KG_HOST,
            port=settings.KG_PORT,
            database=settings.KG_DATABASE,
            user=settings.KG_USER,
            password=settings.KG_PASSWORD
        )
        logger.info(f"Connected to KG storage: {settings.KG_DATABASE}")
        
        return ConnectionResult(
            success=True,
            kg_conn=kg_conn,
            settings=settings
        )
        
    except Exception as e:
        logger.error(f"KG storage connection failed: {e}")
        return ConnectionResult(success=False, error=str(e))


def get_source_connection(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str
) -> ConnectionResult:
    """
    Create source database connection from user-provided credentials.
    
    Args:
        host: Source database host (required - from user input)
        port: Source database port (required - from user input)
        database: Source database name (required - from user input)
        user: Source database user (required - from user input)
        password: Source database password (required - from user input)
    
    Returns:
        ConnectionResult with source_conn or error
    """
    logger.info(f"Creating source database connection to {host}:{port}/{database}...")
    
    try:
        settings = Settings()
        
        # Source database connection (from user input)
        source_conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        logger.info(f"Connected to source database: {database}")
        
        return ConnectionResult(
            success=True,
            source_conn=source_conn,
            settings=settings
        )
        
    except Exception as e:
        logger.error(f"Source database connection failed: {e}")
        return ConnectionResult(success=False, error=str(e))


def get_connections(
    source_host: str,
    source_port: int,
    source_db: str,
    source_user: str,
    source_password: str
) -> ConnectionResult:
    """
    Create both source and KG storage database connections.
    
    Source database credentials come from user input (REQUIRED).
    KG storage credentials come from settings/environment.
    
    Args:
        source_host: Source database host (required - from user input)
        source_port: Source database port (required - from user input)
        source_db: Source database name (required - from user input)
        source_user: Source database user (required - from user input)
        source_password: Source database password (required - from user input)
    
    Returns:
        ConnectionResult with both connections or error
    """
    logger.info("Creating database connections...")
    
    try:
        settings = Settings()
        
        # Source database connection (from user input - REQUIRED)
        source_conn = psycopg2.connect(
            host=source_host,
            port=source_port,
            database=source_db,
            user=source_user,
            password=source_password
        )
        logger.info(f"Connected to source database: {source_db}")
        
        # KG storage connection (from settings/environment)
        kg_conn = psycopg2.connect(
            host=settings.KG_HOST,
            port=settings.KG_PORT,
            database=settings.KG_DATABASE,
            user=settings.KG_USER,
            password=settings.KG_PASSWORD
        )
        logger.info(f"Connected to KG storage: {settings.KG_DATABASE}")
        
        return ConnectionResult(
            success=True,
            source_conn=source_conn,
            kg_conn=kg_conn,
            settings=settings
        )
        
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return ConnectionResult(success=False, error=str(e))


def close_connections(source_conn: Any = None, kg_conn: Any = None) -> None:
    """Close database connections safely"""
    try:
        if source_conn:
            source_conn.close()
            logger.info("Source connection closed")
    except Exception as e:
        logger.warning(f"Error closing source connection: {e}")
    
    try:
        if kg_conn:
            kg_conn.close()
            logger.info("KG connection closed")
    except Exception as e:
        logger.warning(f"Error closing KG connection: {e}")


def check_kg_exists(
    kg_conn: Any,
    source_host: str,
    source_port: int,
    source_db: str
) -> Optional[UUID]:
    """
    Check if a Knowledge Graph already exists for the given source database.
    
    Args:
        kg_conn: KG storage database connection
        source_host: Source database host
        source_port: Source database port
        source_db: Source database name
    
    Returns:
        UUID of existing KG if found, None otherwise
    """
    try:
        db_hash = compute_db_hash(source_host, source_port, source_db)
        kg_repo = KGRepository(kg_conn)
        existing_kg_id = kg_repo.get_kg_by_hash(db_hash)
        
        if existing_kg_id:
            logger.info(f"Found existing KG for {source_db}: {existing_kg_id}")
            return existing_kg_id
        
        logger.info(f"No existing KG found for {source_db}")
        return None
        
    except Exception as e:
        logger.error(f"Error checking for existing KG: {e}")
        return None


def connect_or_build_kg(
    source_host: str,
    source_port: int,
    source_db: str,
    source_user: str,
    source_password: str,
    generate_descriptions: bool = True,
    generate_embeddings: bool = True,
    progress_callback: Optional[ProgressCallback] = None
) -> KGLoadResult:
    """
    Connect to a source database and either load existing KG or build a new one.
    
    This is the main entry point for the UI - handles the complete flow:
    1. Connect to KG storage (from settings)
    2. Connect to source database (from user input)
    3. Check if KG already exists for this source
    4. If exists: load it; if not: build it
    
    Args:
        source_host: Source database host (from user input)
        source_port: Source database port (from user input)
        source_db: Source database name (from user input)
        source_user: Source database user (from user input)
        source_password: Source database password (from user input)
        generate_descriptions: Generate AI descriptions for new KG
        generate_embeddings: Generate embeddings for new KG
        progress_callback: Optional callback for progress updates
    
    Returns:
        KGLoadResult with KG info, connections stored in result
    """
    callback = progress_callback or _default_progress_callback
    
    logger.info("=" * 60)
    logger.info(f"CONNECTING TO DATABASE: {source_db}")
    logger.info("=" * 60)
    
    callback(ProgressUpdate(
        stage="initialization",
        message="Connecting to KG storage...",
        progress=0.0
    ))
    
    # Step 1: Connect to KG storage
    kg_result = get_kg_connection()
    if not kg_result.success:
        return KGLoadResult(success=False, error=f"KG storage connection failed: {kg_result.error}")
    
    kg_conn = kg_result.kg_conn
    settings = kg_result.settings
    
    callback(ProgressUpdate(
        stage="initialization",
        message="Connecting to source database...",
        progress=0.1
    ))
    
    # Step 2: Connect to source database
    source_result = get_source_connection(
        host=source_host,
        port=source_port,
        database=source_db,
        user=source_user,
        password=source_password
    )
    
    if not source_result.success:
        close_connections(kg_conn=kg_conn)
        return KGLoadResult(success=False, error=f"Source database connection failed: {source_result.error}")
    
    source_conn = source_result.source_conn
    
    callback(ProgressUpdate(
        stage="initialization",
        message="Checking for existing Knowledge Graph...",
        progress=0.15
    ))
    
    # Step 3: Check if KG already exists
    existing_kg_id = check_kg_exists(kg_conn, source_host, source_port, source_db)
    
    if existing_kg_id:
        # Load existing KG
        callback(ProgressUpdate(
            stage="loading",
            message="Found existing Knowledge Graph, loading...",
            progress=0.3
        ))
        
        load_result = load_knowledge_graph(
            kg_conn=kg_conn,
            settings=settings,
            kg_id=existing_kg_id
        )
        
        if load_result.success:
            
            callback(ProgressUpdate(
                stage="loading",
                message="Verifying vector embeddings...",
                progress=0.8
            ))
            
            vector_ready = verify_and_fix_vector_store(
                kg_id=existing_kg_id,
                kg_conn=kg_conn,
                settings=settings,
                progress_callback=callback
            )
            
            if not vector_ready:
                logger.warning("Vector store verification failed, but KG loaded successfully")


            callback(ProgressUpdate(
                stage="complete",
                message="Knowledge Graph loaded successfully!",
                progress=1.0,
                details={
                    "tables": load_result.tables_count,
                    "relationships": load_result.relationships_count,
                    "status": "loaded existing",
                    "vector_store": "ready" if vector_ready else "not ready"
                }
            ))
            
            # Return with connection info attached
            return load_result
        
        else:
            # Failed to load, try building new
            logger.warning(f"Failed to load existing KG, will build new: {load_result.error}")
    
    # Step 4: Build new KG
    callback(ProgressUpdate(
        stage="building",
        message="Building new Knowledge Graph...",
        progress=0.2
    ))
    
    build_result = build_knowledge_graph(
        source_conn=source_conn,
        kg_conn=kg_conn,
        settings=settings,
        source_db_name=source_db,
        source_db_host=source_host,
        source_db_port=source_port,
        generate_descriptions=generate_descriptions,
        generate_embeddings=generate_embeddings,
        progress_callback=progress_callback
    )
    
    if build_result.success:
        return KGLoadResult(
            success=True,
            kg_id=build_result.kg_id,
            db_name=source_db,
            tables_count=build_result.tables_count,
            relationships_count=build_result.relationships_count,
            columns_count=build_result.columns_count,
            kg_data=build_result.kg_data
        )
    else:
        close_connections(source_conn, kg_conn)
        return KGLoadResult(success=False, error=build_result.error)

def verify_and_fix_vector_store(
    kg_id: UUID,
    kg_conn: Any,
    settings: Settings,
    progress_callback: Optional[ProgressCallback] = None
) -> bool:
    """
    Verify vector store has embeddings, load from PostgreSQL if missing.
    
    Returns:
        True if vector store is ready, False otherwise
    """
    callback = progress_callback or _default_progress_callback
    
    try:
        
        vector_store = VectorStore(settings.CHROMA_PERSIST_DIR)
        
        callback(ProgressUpdate(
            stage="loading",
            message="Checking vector store...",
            progress=0.85
        ))
        
        # This will automatically load from PostgreSQL if Chroma is empty
        vector_ready = vector_store.ensure_collection_loaded(str(kg_id), kg_conn)
        
        if vector_ready:
            logger.info("Vector store verified and ready")
            return True
        else:
            logger.error("Vector store could not be initialized")
            return False
            
    except Exception as e:
        logger.error(f"Vector store verification failed: {e}", exc_info=True)
        return False

def build_knowledge_graph(
    source_conn: Any,
    kg_conn: Any,
    settings: Settings,
    source_db_name: str,
    source_db_host: str,
    source_db_port: int,
    schema_name: str = "public",
    generate_descriptions: bool = True,
    generate_embeddings: bool = True,
    progress_callback: Optional[ProgressCallback] = None
) -> KGBuildResult:

    callback = progress_callback or _default_progress_callback
    
    logger.info("=" * 60)
    logger.info(f"BUILDING KNOWLEDGE GRAPH: {source_db_name}")
    logger.info("=" * 60)
    
    callback(ProgressUpdate(
        stage="initialization",
        message="Initializing KG Builder...",
        progress=0.0
    ))
    
    try:
        # Initialize OpenAI client
        openai_client = OpenAIClient(
            api_key=settings.OPENAI_API_KEY,
            enable_langfuse=settings.enable_langfuse
        )
        logger.info("OpenAI client initialized")
        
        callback(ProgressUpdate(
            stage="initialization",
            message="OpenAI client ready",
            progress=0.1
        ))
        
        # Initialize builder
        builder = KGBuilder(
            source_conn=source_conn,
            kg_conn=kg_conn,
            openai_client=openai_client,
            settings=settings
        )
        
        callback(ProgressUpdate(
            stage="schema_extraction",
            message="Extracting database schema...",
            progress=0.2
        ))
        
        # Build KG
        kg = builder.build_kg(
            source_db_name=source_db_name,
            source_db_host=source_db_host,
            source_db_port=source_db_port,
            schema_name=schema_name,
            generate_descriptions=generate_descriptions,
            generate_embeddings=generate_embeddings
        )
        
        if kg:
            callback(ProgressUpdate(
                stage="complete",
                message="Knowledge Graph built successfully!",
                progress=1.0,
                details={
                    "tables": len(kg.tables),
                    "relationships": len(kg.relationships)
                }
            ))
            
            total_cols = sum(len(t.columns) for t in kg.tables.values())
            
            # Build KG data for UI visualization
            kg_data = {
                "tables": {
                    name: {
                        "description": t.description,
                        "domain": t.business_domain,
                        "columns": {
                            c: {
                                "type": col.data_type,
                                "pk": col.is_primary_key,
                                "fk": col.is_foreign_key,
                                "description": col.description
                            }
                            for c, col in t.columns.items()
                        }
                    }
                    for name, t in kg.tables.items()
                },
                "relationships": [
                    {
                        "from": r.from_table_name,
                        "to": r.to_table_name,
                        "from_column": r.from_column,
                        "to_column": r.to_column
                    }
                    for r in kg.relationships
                ]
            }
            
            logger.info(f"KG Build Complete: {len(kg.tables)} tables, {len(kg.relationships)} relationships")
            
            return KGBuildResult(
                success=True,
                kg_id=kg.kg_id,
                tables_count=len(kg.tables),
                relationships_count=len(kg.relationships),
                columns_count=total_cols,
                message=f"Successfully built KG with {len(kg.tables)} tables",
                kg_data=kg_data
            )
        
        return KGBuildResult(
            success=False,
            error="Failed to build Knowledge Graph"
        )
        
    except Exception as e:
        logger.error(f"KG build failed: {e}", exc_info=True)
        callback(ProgressUpdate(
            stage="error",
            message=f"Build failed: {str(e)}",
            progress=0.0
        ))
        return KGBuildResult(success=False, error=str(e))


def load_knowledge_graph(
    kg_conn: Any,
    settings: Settings,
    kg_id: Optional[UUID] = None,
    source_db_hash: Optional[str] = None,
    source_db_host: Optional[str] = None,
    source_db_port: Optional[int] = None,
    source_db_name: Optional[str] = None
) -> KGLoadResult:
    
    logger.info("Loading Knowledge Graph...")
    
    try:
        kg_manager = KGManager(kg_conn, settings.CHROMA_PERSIST_DIR)
        
        # Determine KG to load
        target_kg_id = kg_id
        
        if not target_kg_id and source_db_hash:
            target_kg_id = KGRepository(kg_conn).get_kg_by_hash(source_db_hash)
        
        if not target_kg_id and all([source_db_host, source_db_port, source_db_name]):
            hash_str = f"{source_db_host}:{source_db_port}:{source_db_name}"
            computed_hash = hashlib.sha256(hash_str.encode()).hexdigest()
            target_kg_id = KGRepository(kg_conn).get_kg_by_hash(computed_hash)
        
        if not target_kg_id:
            return KGLoadResult(
                success=False,
                error="No Knowledge Graph found matching criteria"
            )
        
        # Load KG
        kg = kg_manager.load_kg(target_kg_id)
        
        if kg:
            total_cols = sum(len(t.columns) for t in kg.tables.values())
            
            # Build KG data for UI
            kg_data = {
                "tables": {
                    name: {
                        "description": t.description,
                        "domain": t.business_domain,
                        "columns": {
                            c: {
                                "type": col.data_type,
                                "pk": col.is_primary_key,
                                "fk": col.is_foreign_key,
                                "description": col.description
                            }
                            for c, col in t.columns.items()
                        }
                    }
                    for name, t in kg.tables.items()
                },
                "relationships": [
                    {
                        "from": r.from_table_name,
                        "to": r.to_table_name,
                        "from_column": r.from_column,
                        "to_column": r.to_column
                    }
                    for r in kg.relationships
                ]
            }
            
            logger.info(f"KG Loaded: {kg.kg_id} ({len(kg.tables)} tables)")
            
            return KGLoadResult(
                success=True,
                kg_id=kg.kg_id,
                db_name=kg.source_db_name,
                tables_count=len(kg.tables),
                relationships_count=len(kg.relationships),
                columns_count=total_cols,
                kg_data=kg_data
            )
        
        return KGLoadResult(success=False, error="Failed to load Knowledge Graph")
        
    except Exception as e:
        logger.error(f"KG load failed: {e}", exc_info=True)
        return KGLoadResult(success=False, error=str(e))


def list_knowledge_graphs(kg_conn: Any) -> List[KGListItem]:
    
    logger.info("Listing Knowledge Graphs...")
    
    try:
        query = """
            SELECT 
                m.kg_id,
                m.source_db_name,
                m.source_db_host,
                m.source_db_port,
                m.status,
                m.created_at,
                m.last_updated,
                COUNT(DISTINCT t.table_id) as tables_count
            FROM kg_metadata m
            LEFT JOIN kg_tables t ON m.kg_id = t.kg_id
            GROUP BY m.kg_id, m.source_db_name, m.source_db_host, 
                     m.source_db_port, m.status, m.created_at, m.last_updated
            ORDER BY m.last_updated DESC
        """
        
        with kg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            results = cur.fetchall()
        
        items = []
        for row in results:
            items.append(KGListItem(
                kg_id=UUID(row['kg_id']),
                db_name=row['source_db_name'],
                db_host=row['source_db_host'],
                db_port=row['source_db_port'],
                status=row['status'],
                tables_count=row['tables_count'] or 0,
                created_at=row['created_at'],
                last_updated=row['last_updated']
            ))
        
        logger.info(f"Found {len(items)} Knowledge Graphs")
        return items
        
    except Exception as e:
        logger.error(f"Failed to list KGs: {e}")
        return []


_agent_service_cache: Dict[str, AgentService] = {}


def get_agent_service(
    kg_conn: Any,
    source_conn: Any,
    settings: Settings
) -> AgentService:
    """
    Get or create an AgentService instance.
    
    Args:
        kg_conn: KG storage connection
        source_conn: Source database connection
        settings: Application settings
    
    Returns:
        AgentService instance
    """
    # Use a simple cache key
    cache_key = f"{id(kg_conn)}_{id(source_conn)}"
    
    if cache_key not in _agent_service_cache:
        logger.info("Initializing Agent Service...")
        
        openai_client = OpenAIClient(
            api_key=settings.OPENAI_API_KEY,
            enable_langfuse=settings.enable_langfuse
        )
        
        kg_manager = KGManager(kg_conn, settings.CHROMA_PERSIST_DIR)
        
        _agent_service_cache[cache_key] = AgentService(
            kg_manager=kg_manager,
            openai_client=openai_client,
            source_db_conn=source_conn,
            kg_conn=kg_conn
        )
        
        logger.info("Agent Service initialized")
    
    return _agent_service_cache[cache_key]


def process_query(
    agent_service: AgentService,
    kg_id: UUID,
    user_query: str,
    clarifications: Optional[Dict[str, str]] = None,
    progress_callback: Optional[ProgressCallback] = None
) -> QueryResult:
    """
        Process a natural language query.
    """
    callback = progress_callback or _default_progress_callback
    
    logger.info("=" * 60)
    logger.info(f"PROCESSING QUERY: {user_query}")
    logger.info("=" * 60)
    
    callback(ProgressUpdate(
        stage="query_analysis",
        message="Analyzing query...",
        progress=0.1
    ))
    
    try:
        callback(ProgressUpdate(
            stage="schema_selection",
            message="Selecting relevant tables...",
            progress=0.3
        ))
        
        # Execute query through agent service
        response = agent_service.query(
            user_query=user_query,
            kg_id=kg_id,
            clarifications=clarifications
        )
        
        # Check for clarification needed
        if response.get("needs_clarification"):
            callback(ProgressUpdate(
                stage="clarification",
                message="Clarification needed",
                progress=0.5
            ))
            
            return QueryResult(
                success=False,
                needs_clarification=True,
                clarification_request=response.get("clarification_request")
            )
        
        callback(ProgressUpdate(
            stage="sql_generation",
            message="Generating SQL...",
            progress=0.5
        ))
        
        callback(ProgressUpdate(
            stage="execution",
            message="Executing query...",
            progress=0.8
        ))
        
        if response.get("success"):
            callback(ProgressUpdate(
                stage="complete",
                message="Query completed successfully!",
                progress=1.0
            ))
            
            return QueryResult(
                success=True,
                data=response.get("data"),
                sql=response.get("sql"),
                explanation=response.get("explanation"),
                metadata=response.get("metadata", {}),
                trace_id=response.get("trace_id")
            )
        else:
            callback(ProgressUpdate(
                stage="error",
                message=f"Query failed: {response.get('error', 'Unknown error')}",
                progress=0.0
            ))
            
            return QueryResult(
                success=False,
                error=response.get("error"),
                error_category=response.get("error_category"),
                sql=response.get("sql_attempted"),
                metadata=response.get("metadata", {}),
                trace_id=response.get("trace_id")
            )
            
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        callback(ProgressUpdate(
            stage="error",
            message=f"Processing failed: {str(e)}",
            progress=0.0
        ))
        return QueryResult(success=False, error=str(e), trace_id=response.get("trace_id"))


def submit_feedback(
    agent_service: AgentService,
    query_log_id: Union[UUID, str],
    feedback: str,
    rating: Optional[int] = None
) -> FeedbackResult:
    """
        Submit user feedback for a query result.
    """
    logger.info(f"Submitting feedback for query {query_log_id}")
    
    try:
        if isinstance(query_log_id, str):
            query_log_id = UUID(query_log_id)
            
        result = agent_service.submit_feedback(
            query_log_id=query_log_id,
            feedback=feedback,
            rating=rating
        )
        
        if result.get("success"):
            logger.info("Feedback submitted successfully")
            return FeedbackResult(success=True, lesson_extracted=result.get("lesson_extracted", False))
        else:
            return FeedbackResult(success=False, error=result.get("error"))
            
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        return FeedbackResult(success=False, error=str(e))


def get_log_file_path() -> Path:
    """Get the path to the log file"""
    return LOG_FILE


def clear_agent_service_cache() -> None:
    """Clear the agent service cache"""
    global _agent_service_cache
    _agent_service_cache.clear()
    logger.info("Agent service cache cleared")


def compute_db_hash(host: str, port: int, database: str) -> str:
    """Compute database hash for KG lookup"""
    hash_str = f"{host}:{port}:{database}"
    return hashlib.sha256(hash_str.encode()).hexdigest()


if __name__ == "__main__":
    print("Text2SQL Agent - Main API Layer")
    print("=" * 40)
    print(f"Log file: {LOG_FILE}")
    print("\nAvailable functions:")
    print("  - get_connections()")
    print("  - build_knowledge_graph()")
    print("  - load_knowledge_graph()")
    print("  - list_knowledge_graphs()")
    print("  - process_query()")
    print("  - submit_feedback()")
    print("\nImport this module to use the API.")
