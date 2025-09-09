"""
Example usage of the Bio Data Vector Store.

This script demonstrates how to use the vector store for bio data matching.
Run this after setting up the vector store and adding some bio data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_vector_store.setup_vector_store import VectorStoreSetup


def main():
    """Demonstrate vector store usage."""
    print("Bio Data Vector Store Example")
    print("=" * 40)
    
    # Initialize vector store
    print("1. Initializing vector store...")
    setup = VectorStoreSetup()
    
    # Check if setup is successful
    if not setup.setup_vector_store():
        print("Failed to setup vector store. Check dependencies.")
        return
    
    print("✓ Vector store initialized")
    
    # Get current statistics
    print("\n2. Current vector store statistics:")
    stats = setup.get_vector_store_stats()
    if "error" not in stats:
        print(f"   Total vectors: {stats.get('total_vectors', 0)}")
        print(f"   My bio data: {stats.get('my_biodata_count', 0)}")
        print(f"   People bio data: {stats.get('ppl_biodata_count', 0)}")
        print(f"   Index size: {stats.get('total_size_mb', 0):.2f} MB")
    else:
        print(f"   Error: {stats['error']}")
    
    # Example: Add bio data from PDF (if file exists)
    print("\n3. Example: Adding bio data from PDF...")
    pdf_path = "data/raw/ppl_biodata/example.pdf"
    if Path(pdf_path).exists():
        bio_data_id = setup.process_and_add_pdf(pdf_path, "ppl_biodata")
        if bio_data_id:
            print(f"✓ Added bio data with ID: {bio_data_id}")
        else:
            print("✗ Failed to add PDF")
    else:
        print(f"   PDF file not found: {pdf_path}")
        print("   To test this feature, place a PDF file at the above path")
    
    # Example: Add bio data from JSON (if file exists)
    print("\n4. Example: Adding bio data from JSON...")
    json_path = "data/processed/ppl_biodata/example.json"
    if Path(json_path).exists():
        bio_data_id = setup.add_bio_data_from_json(json_path)
        if bio_data_id:
            print(f"✓ Added bio data with ID: {bio_data_id}")
        else:
            print("✗ Failed to add JSON")
    else:
        print(f"   JSON file not found: {json_path}")
        print("   To test this feature, first extract bio data using the bio_extractor")
    
    # List existing bio data
    print("\n5. Listing existing bio data...")
    bio_data_list = setup.list_bio_data("ppl_biodata")
    if bio_data_list:
        print(f"   Found {len(bio_data_list)} bio data entries:")
        for i, bio_data in enumerate(bio_data_list[:5], 1):  # Show first 5
            name = bio_data.get('personal_info', {}).get('name', 'Unknown')
            print(f"   {i}. {bio_data['bio_data_id']} - {name}")
        
        if len(bio_data_list) > 5:
            print(f"   ... and {len(bio_data_list) - 5} more")
    else:
        print("   No bio data found")
    
    # Example: Search for similar bio data
    print("\n6. Example: Searching for similar bio data...")
    
    # Search using text query
    query_text = "Software engineer interested in hiking and photography"
    print(f"   Searching for: '{query_text}'")
    
    results = setup.search_similar_bio_data(
        query_text=query_text,
        target_bio_type="ppl_biodata",
        k=5,
        similarity_threshold=0.5  # Lower threshold for demo
    )
    
    if "error" not in results:
        print(f"   Found {results['total_results']} similar bio data entries:")
        print(f"   Search time: {results['search_time_ms']:.2f}ms")
        
        for i, result in enumerate(results['results'], 1):
            name = result.get('personal_info', {}).get('name', 'Unknown')
            score = result['similarity_score']
            print(f"   {i}. {name} (similarity: {score:.3f})")
    else:
        print(f"   Search error: {results['error']}")
    
    # Example: Search using bio data ID (if available)
    if bio_data_list:
        print("\n7. Example: Searching using bio data ID...")
        query_id = bio_data_list[0]['bio_data_id']
        print(f"   Using bio data ID: {query_id}")
        
        results = setup.search_similar_bio_data(
            query_bio_data_id=query_id,
            target_bio_type="ppl_biodata",
            k=3
        )
        
        if "error" not in results:
            print(f"   Found {results['total_results']} similar entries")
            for i, result in enumerate(results['results'], 1):
                name = result.get('personal_info', {}).get('name', 'Unknown')
                score = result['similarity_score']
                print(f"   {i}. {name} (similarity: {score:.3f})")
        else:
            print(f"   Search error: {results['error']}")
    
    # Example: Create backup
    print("\n8. Example: Creating backup...")
    try:
        backup_path = setup.backup_vector_store()
        print(f"✓ Created backup at: {backup_path}")
    except Exception as e:
        print(f"✗ Backup failed: {e}")
    
    # Performance example
    print("\n9. Performance optimization...")
    try:
        setup.optimize_vector_store()
        print("✓ Vector store optimized")
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
    
    # Final statistics
    print("\n10. Final statistics:")
    final_stats = setup.get_vector_store_stats()
    if "error" not in final_stats:
        print(f"    Total vectors: {final_stats.get('total_vectors', 0)}")
        print(f"    Total searches: {final_stats.get('total_searches', 0)}")
        avg_time = final_stats.get('average_search_time_ms')
        if avg_time:
            print(f"    Average search time: {avg_time:.2f}ms")
    
    # Clean up
    setup.close()
    print("\n✓ Vector store closed")
    
    print("\nExample completed successfully!")
    print("\nNext steps:")
    print("- Add your own bio data using setup.process_and_add_pdf()")
    print("- Experiment with different search queries")
    print("- Adjust configuration in data_vector_store/config/vector_store_config.yaml")
    print("- Try different embedding models for better results")


if __name__ == "__main__":
    main()
