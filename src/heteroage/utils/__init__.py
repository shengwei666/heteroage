from .bio_utils import (
    load_pathway_definitions as load_hallmark_dict,  # Alias for backward compatibility
    construct_biosparse_topology as create_tiered_hallmark_mask, # Alias: maps CLI call to new function
    setup_logger
)

# Export these names to be accessible via 'from .utils import ...'
__all__ = [
    'load_hallmark_dict',
    'create_tiered_hallmark_mask', 
    'setup_logger'
]