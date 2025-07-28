import pandas as pd
from typing import Dict, List, Union, Optional, Callable, Any
import logging
import time

try:
    spark
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
try:
    from pyspark.sql import SparkSession, DataFrame
    import pyspark.sql.functions as F
    from pyspark.sql.window import Window
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = None

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def wrapped_batch_processing(
    spark_df,
    process_batch_fn: Callable,
    chunk_size: int = 100000,
    inner_batch_size: int = 10000,
    output_table_prefix: str = "batch_",
    output_schema: Optional[str] = None,
    continue_on_error: bool = True,
    order_by_col: Optional[Union[str, List[str]]] = None,
    start_chunk_number: int = 1
) -> Dict[str, Any]:
    """
    Simple wrapped batch processing - splits DataFrame into chunks and processes each.
    
    Parameters:
    -----------
    spark_df : pyspark.sql.DataFrame
        Source Spark DataFrame to process
    process_batch_fn : callable
        Function to apply to each inner batch
    chunk_size : int, default=100000
        Number of rows per chunk/table (100K by default)
    inner_batch_size : int, default=10000
        Batch size for processing within each chunk
    output_table_prefix : str, default="batch_"
        Prefix for output table names (will be batch_1, batch_2, etc.)
    output_schema : str, optional
        Schema name for output tables
    continue_on_error : bool, default=True
        Whether to continue if a chunk fails
    order_by_col : str or list, optional
        Column(s) to order by when adding row_number for chunking.
        If None, uses monotonically_increasing_id() (less deterministic).
        Recommended to provide for consistent, reproducible chunking.
    start_chunk_number : int, default=1
        Starting number for chunk table names. Useful for resuming processing.
        E.g., if set to 13, tables will be named batch_13, batch_14, etc.
        
    Returns:
    --------
    Dict[str, Any]
        Processing statistics
    """
    if not PYSPARK_AVAILABLE:
        raise RuntimeError("PySpark is required")
    
    start_time = time.time()
    total_rows = spark_df.count()
    total_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division
    
    print(f"🚀 Starting simple wrapped batch processing")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Chunk size: {chunk_size:,}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Inner batch size: {inner_batch_size:,}")
    
    successful_chunks = 0
    failed_chunks = 0
    chunk_details = []
    
    # Add row numbers for chunking using row_number() window function for consecutive numbering
    if order_by_col:
        if isinstance(order_by_col, str):
            order_cols = [order_by_col]
        else:
            order_cols = order_by_col
        window_spec = Window.orderBy(*[F.col(c) for c in order_cols])
    else:
        window_spec = Window.orderBy(F.monotonically_increasing_id())
    
    spark_df_with_row_num = spark_df.withColumn("_row_number", F.row_number().over(window_spec) - 1)  # 0-based indexing
    
    for i in range(total_chunks):
        chunk_start_time = time.time()
        print(f"i value is {i}")
        chunk_table_name = f"{output_table_prefix}{start_chunk_number + i}"
        
        try:
            print(f"\n📦 Processing chunk {i + 1}/{total_chunks}")
            print(f"   Target table: {chunk_table_name}")
            
            # Extract chunk based on row range
            start_row = i * chunk_size
            end_row = min((i + 1) * chunk_size, total_rows) - 1  # Inclusive end for row_number
            
            # Get the chunk using consecutive row numbers
            chunk_df = (spark_df_with_row_num
                       .filter((F.col("_row_number") >= start_row) & (F.col("_row_number") <= end_row))
                       .drop("_row_number"))
            
            chunk_row_count = chunk_df.count()
            print(f"   Chunk rows: {chunk_row_count:,}")
            
            # Process the chunk
            process_dataframe_in_batches(
                spark_df=chunk_df,
                batch_size=inner_batch_size,
                process_batch_fn=process_batch_fn,
                write_output=True,
                output_table_name=chunk_table_name,
                output_schema=output_schema,
                continue_on_error=continue_on_error,
                order_by_col=order_by_col,
                start_from_batch = 0
            )
            
            chunk_processing_time = time.time() - chunk_start_time
            print(f"   ✅ Chunk {i + 1} completed in {chunk_processing_time:.2f}s")
            
            successful_chunks += 1
            chunk_details.append({
                'chunk_number': start_chunk_number + i,
                'table_name': chunk_table_name,
                'rows': chunk_row_count,
                'processing_time': chunk_processing_time,
                'status': 'success'
            })
            
        except Exception as e:
            chunk_processing_time = time.time() - chunk_start_time
            error_msg = f"Error processing chunk {i + 1}: {str(e)}"
            print(f"   ❌ {error_msg}")
            
            failed_chunks += 1
            chunk_details.append({
                'chunk_number': start_chunk_number + i,
                'table_name': chunk_table_name,
                'rows': 0,
                'processing_time': chunk_processing_time,
                'status': 'failed',
                'error': str(e)
            })
            
            if continue_on_error:
                print(f"   ⏭️ Continuing to next chunk...")
                continue
            else:
                raise RuntimeError(error_msg) from e
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Successful: {successful_chunks}")
    print(f"   Failed: {failed_chunks}")
    print(f"   Total time: {total_time:.2f} seconds")
    
    return {
        'total_chunks': total_chunks,
        'successful_chunks': successful_chunks,
        'failed_chunks': failed_chunks,
        'total_time': total_time,
        'chunk_details': chunk_details
    }

def create_dynamic_field_mapping_for_row(
    row_data: Union[pd.Series, dict], 
    field_configs: Dict[str, Union[str, List[str]]]
) -> Dict[str, List[str]]:
    """
    Create a dynamic field mapping for a single row by filtering out null/None values from numbered fields.
    
    Args:
        row_data: Single row of data (pandas Series or dict)
        field_configs: Dictionary mapping PII tags to field patterns
                      e.g., {'<FIRST_NAME>': 'FIRSTNAME_', '<LAST_NAME>': 'SURNAME_'}
    
    Returns:
        Dictionary with PII tags mapped to lists of valid (non-null) field names for this specific row.
    
    Example:
        field_configs = {
            '<FIRST_NAME>': 'FIRST_NAME_',
            '<LAST_NAME>': 'SURNAME_'
        }
        
        # Sample DataFrame
        data = pd.DataFrame({
            'FIRST_NAME_FATHER': ['John', 'Kate'],
            'FIRST_NAME_MOTHER': ['Jane', None],
            'SURNAME_1': [None, 'Smith'],
            'SURNAME_2': ['Doe', 'Smith']
        })
        
        # For first row (index 0):
        # {
        #     '<FIRST_NAME>': ['FIRST_NAME_FATHER', 'FIRST_NAME_MOTHER'],
        #     '<LAST_NAME>': ['SURNAME_2']
        # }
        
        # For second row (index 1):
        # {
        #     '<FIRST_NAME>': ['FIRST_NAME_FATHER'],
        #     '<LAST_NAME>': ['SURNAME_1', 'SURNAME_2']
        # }
    """
    field_mapping = {}
    
    # Convert to dict if it's a pandas Series
    if hasattr(row_data, 'to_dict'):
        row_dict = row_data.to_dict()
    else:
        row_dict = row_data
    
    for pii_tag, field_pattern in field_configs.items():
        valid_fields = []
        
        # Handle special cases where field_pattern is a list (like phone numbers)
        if isinstance(field_pattern, list):
            for field_name in field_pattern:
                if field_name in row_dict:
                    value = row_dict[field_name]
                    if pd.notna(value) and value != 'None' and value != '':
                        valid_fields.append(field_name)
        else:
            # Find all columns matching the pattern in this specific row
            matching_fields = [col for col in row_dict.keys() if col.startswith(field_pattern)]
            
            # Check each matching field for non-null values in this row
            for field_name in matching_fields:
                value = row_dict[field_name]
                # Check if this specific row has a valid value for this field
                if pd.notna(value) and value != 'None' and value != '':
                    valid_fields.append(field_name)
        
        if valid_fields:
            field_mapping[pii_tag] = sorted(valid_fields)  # Sort for consistency
    
    return field_mapping

def process_dataframe_in_batches(
    spark_df,  # SparkDataFrame when PySpark is available
    batch_size: int = 100,
    process_batch_fn: Optional[Callable] = None,
    write_output: bool = True,
    output_table_name: Optional[str] = None,
    output_schema: Optional[str] = None,
    partition_col: Optional[str] = None,
    select_cols: Optional[List[str]] = None,
    order_by_col: Optional[Union[str, List[str]]] = None,
    start_from_batch: int = 0,
    max_batches: Optional[int] = None,
    continue_on_error: bool = False
):
    """
    Process a large DataFrame in batches with flexible processing functions.
    
    Parameters:
    -----------
    spark_df : pyspark.sql.DataFrame
        Source Spark DataFrame to process
    batch_size : int, default=100
        Number of rows per batch
    process_batch_fn : callable, default=None
        Function to apply to each batch. It should accept a DataFrame (or pandas DataFrame)
        and return a processed DataFrame. If None, no processing is performed.
        Function signature: fn(batch_df, batch_num, total_batches) -> processed_df
    write_output : bool, default=True
        Whether to write the processed batches to a table
    output_table_name : str, optional
        Target table name when write_output=True
    output_schema : str, optional
        Target schema name when write_output=True
    partition_col : str, optional
        Column to use for partitioning when extracting batches (defaults to adding a row_number)
    select_cols : list, optional
        Columns to select from the source DataFrame
    order_by_col : str or list, optional
        Column(s) to order by when adding row_number for batching
    start_from_batch : int, default=0
        Batch number to start processing from (0-indexed)
    max_batches : int, optional
        Maximum number of batches to process. If None, process all batches.
    continue_on_error : bool, default=False
        Whether to continue processing next batches if an error occurs
        
    Returns:
    --------
    Optional[pyspark.sql.DataFrame]
        Combined result of all batches if write_output=False, otherwise None
        
    Raises:
    -------
    RuntimeError
        If PySpark is not available
    ValueError
        If required parameters are missing or invalid
    RuntimeError
        If processing fails and continue_on_error=False
        
    Examples:
    ---------
    >>> # Simple batch processing without custom function
    >>> result = process_dataframe_in_batches(
    ...     spark_df=my_df,
    ...     batch_size=50,
    ...     write_output=False
    ... )
    
    >>> # Custom processing with error handling
    >>> def my_processor(batch_df, batch_num, total_batches):
    ...     # Custom processing logic here
    ...     return batch_df.withColumn("processed", F.lit(True))
    ...
    >>> process_dataframe_in_batches(
    ...     spark_df=my_df,
    ...     batch_size=100,
    ...     process_batch_fn=my_processor,
    ...     output_table_name="processed_data",
    ...     continue_on_error=True
    ... )
    """
    # Check if PySpark is available
    if not PYSPARK_AVAILABLE:
        raise RuntimeError(
            "PySpark is not available. Please install PySpark using: pip install pyspark"
        )
    # Validation
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    if start_from_batch < 0:
        raise ValueError("start_from_batch must be non-negative")
    
    if write_output and not output_table_name:
        raise ValueError("output_table_name is required when write_output=True")
    
    # Select specific columns if specified
    if select_cols:
        working_df = spark_df.select(*select_cols)
    else:
        working_df = spark_df
    
    # Add batch partitioning column if not provided
    if partition_col is None:
        # Create window specification for row numbering
        if order_by_col:
            if isinstance(order_by_col, str):
                order_cols = [order_by_col]
            else:
                order_cols = order_by_col
            window_spec = Window.orderBy(*[F.col(c) for c in order_cols])
        else:
            window_spec = Window.orderBy(F.lit(1))  # Default ordering
        
        # Add row number and batch number
        working_df = working_df.withColumn("_row_num", F.row_number().over(window_spec))
        working_df = working_df.withColumn("_batch_num", 
                                         ((F.col("_row_num") - 1) / batch_size).cast("int"))
        partition_col = "_batch_num"
    
    # Calculate total number of rows and batches
    total_rows = working_df.count()
    total_batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division
    
    logger.info(f"Processing {total_rows} rows in {total_batches} batches of size {batch_size}")
    
    # Determine batch range to process
    end_batch = total_batches
    if max_batches:
        end_batch = min(start_from_batch + max_batches, total_batches)
    
    processed_batches = []
    successful_batches = 0
    failed_batches = 0
    
    # Process each batch
    for batch_num in range(start_from_batch, end_batch):
        try:
            logger.info(f"Processing batch {batch_num + 1}/{total_batches}")
            print(f"batch_num: {batch_num}")
            print(f"total_batches: {total_batches}")
            print(f"Processing batch {batch_num + 1}/{total_batches}")
            
            # Extract current batch
            batch_df = working_df.filter(F.col(partition_col) == batch_num)
            
            # Remove internal columns if they were added
            if partition_col in ["_batch_num", "_row_num"]:
                batch_df = batch_df.drop("_batch_num", "_row_num")
            
            # Apply custom processing function if provided
            if process_batch_fn:
                processed_batch = process_batch_fn(batch_df, batch_num, total_batches)
                if processed_batch is None:
                    logger.warning(f"Batch {batch_num} processing returned None")
                    continue
            else:
                processed_batch = batch_df
            
            # Handle output
            if write_output:
                # Construct full table name
                full_table_name = output_table_name
                if output_schema:
                    full_table_name = f"{output_schema}.{output_table_name}"
                
                # Write batch to table
                if batch_num == start_from_batch:
                    # First batch - create/overwrite table
                    processed_batch.write.mode("overwrite").saveAsTable(full_table_name)
                    logger.info(f"Created table {full_table_name} with first batch")
                else:
                    # Subsequent batches - append
                    processed_batch.write.mode("append").saveAsTable(full_table_name)
                    logger.debug(f"Appended batch {batch_num} to {full_table_name}")
            else:
                # Collect results for return
                processed_batches.append(processed_batch)
            
            successful_batches += 1
            
        except Exception as e:
            failed_batches += 1
            error_msg = f"Error processing batch {batch_num}: {str(e)}"
            logger.error(error_msg)
            
            if continue_on_error:
                logger.warning(f"Continuing to next batch due to continue_on_error=True")
                continue
            else:
                raise RuntimeError(error_msg) from e
    
    # Log summary
    logger.info(f"Batch processing complete. Successful: {successful_batches}, Failed: {failed_batches}")
    
    # If writing to table and order_by_col is specified, ensure final table is properly ordered
    if write_output and order_by_col and successful_batches > 0:
        full_table_name = output_table_name
        if output_schema:
            full_table_name = f"{output_schema}.{output_table_name}"
        
        logger.info(f"Reordering final table {full_table_name} by {order_by_col}")
        print(f"🔄 Reordering final table by {order_by_col}")
        
        # Read the table, order it, and rewrite it
        final_df = spark.table(full_table_name)
        if isinstance(order_by_col, str):
            ordered_df = final_df.orderBy(F.col(order_by_col))
        else:
            ordered_df = final_df.orderBy(*[F.col(c) for c in order_by_col])
        
        # Rewrite the table with proper ordering
        ordered_df.write.mode("overwrite").saveAsTable(full_table_name)
        logger.info(f"Final table {full_table_name} reordered successfully")
        print(f"✅ Final table reordered successfully")
    
    # Return combined results if not writing to table
    if not write_output and processed_batches:
        logger.info("Combining all processed batches into single DataFrame")
        # Union all processed batches
        combined_df = processed_batches[0]
        for batch in processed_batches[1:]:
            combined_df = combined_df.union(batch)
        return combined_df
    
    return None

def add_sequence_id(
    spark_df,
    order_by_col: Optional[Union[str, List[str]]] = None,
    sequence_col_name: str = "sequence_id"
):
    """
    Add a sequence_id column to a Spark DataFrame based on ordering.
    
    This is particularly useful for large-scale batch processing where you need:
    - Deterministic ordering across processing runs
    - Restart capability if processing fails partway through
    - Clear data lineage and debugging
    
    Parameters:
    -----------
    spark_df : pyspark.sql.DataFrame
        Source Spark DataFrame
    order_by_col : str or list, optional
        Column(s) to order by when creating sequence numbers.
        If None, uses monotonically_increasing_id() (less deterministic).
        Recommended to provide for consistent, reproducible sequencing.
    sequence_col_name : str, default="sequence_id"
        Name of the sequence column to add
        
    Returns:
    --------
    pyspark.sql.DataFrame
        DataFrame with added sequence_id column (1-indexed)
        
    Examples:
    ---------
    >>> # Add sequence based on existing ID column
    >>> df_with_seq = add_sequence_id(df, order_by_col='original_id')
    
    >>> # Add sequence based on multiple columns
    >>> df_with_seq = add_sequence_id(df, order_by_col=['timestamp', 'id'])
    
    >>> # For 1.7M records processing with restart capability
    >>> df_with_seq = add_sequence_id(transcript_df, order_by_col='original_order')
    >>> # Now sequence_id goes from 1 to 1,700,000 in deterministic order
    """
    if not PYSPARK_AVAILABLE:
        raise RuntimeError("PySpark is required")
    
    # Create window specification for sequence numbering
    if order_by_col:
        if isinstance(order_by_col, str):
            order_cols = [order_by_col]
        else:
            order_cols = order_by_col
        window_spec = Window.orderBy(*[F.col(c) for c in order_cols])
        print(f"📊 Adding {sequence_col_name} ordered by: {order_cols}")
    else:
        window_spec = Window.orderBy(F.monotonically_increasing_id())
        print(f"⚠️  Adding {sequence_col_name} with non-deterministic ordering (no order_by_col specified)")
    
    # Add sequence_id column (1-indexed for human readability)
    df_with_sequence = spark_df.withColumn(sequence_col_name, F.row_number().over(window_spec))
    
    total_rows = df_with_sequence.count()
    print(f"✅ Added {sequence_col_name} column: 1 to {total_rows:,}")
    
    return df_with_sequence