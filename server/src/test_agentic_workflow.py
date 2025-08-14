#!/usr/bin/env python3
"""
Test script for the new agentic RAG workflow.
This script tests various scenarios to ensure the agentic behavior works correctly.
"""

import asyncio
import json
import os
from dotenv import load_dotenv
from langgraph_workflow import process_rag_query
from logger import logger

load_dotenv()

async def test_agentic_workflow():
    """Test the agentic RAG workflow with different query scenarios"""
    
    print("🤖 Testing Agentic RAG Workflow")
    print("=" * 50)
    
    # Test cases with expected behavior
    test_cases = [
        {
            "name": "Simple Query (should find answer quickly)",
            "query": "What is machine learning?",
            "expected_iterations": 1
        },
        {
            "name": "Specific Technical Query (might need refinement)",
            "query": "How does gradient descent optimization work in neural networks?",
            "expected_iterations": 1
        },
        {
            "name": "Vague Query (likely needs refinement)",
            "query": "AI stuff",
            "expected_iterations": 2
        },
        {
            "name": "Very Specific Query (might not find exact match)",
            "query": "What are the specific hyperparameter tuning strategies for BERT models trained on biomedical text with limited GPU memory?",
            "expected_iterations": 2
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print(f"Query: '{test_case['query']}'")
        print("-" * 40)
        
        try:
            # Run the agentic workflow
            result = await process_rag_query(test_case['query'])
            
            # Extract metadata
            metadata = result.get('metadata', {})
            iterations = metadata.get('iterations', 0)
            is_adequate = metadata.get('is_answer_adequate', False)
            query_history = metadata.get('query_history', [])
            original_query = metadata.get('original_query', '')
            final_query = metadata.get('final_query', '')
            evaluation_details = metadata.get('evaluation_details', {})
            
            print(f"✅ Completed in {iterations + 1} iteration(s)")
            print(f"📊 Answer adequate: {is_adequate}")
            print(f"🔍 Original query: '{original_query}'")
            if final_query != original_query:
                print(f"🔄 Final query: '{final_query}'")
            
            if len(query_history) > 1:
                print(f"📚 Query evolution:")
                for j, q in enumerate(query_history):
                    print(f"   {j + 1}. {q}")
            
            # Show evaluation reasoning if available
            if evaluation_details.get('reasoning'):
                print(f"🧠 Evaluation: {evaluation_details['reasoning']}")
            
            # Show summary length
            summary = result.get('summary', '')
            print(f"📄 Summary length: {len(summary)} characters")
            
            # Show first part of summary
            if summary:
                preview = summary[:200] + "..." if len(summary) > 200 else summary
                print(f"📖 Summary preview: {preview}")
            
            # Store results for analysis
            results.append({
                'test_name': test_case['name'],
                'query': test_case['query'],
                'iterations': iterations + 1,
                'is_adequate': is_adequate,
                'query_history': query_history,
                'summary_length': len(summary),
                'evaluation_details': evaluation_details,
                'error': result.get('error')
            })
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append({
                'test_name': test_case['name'],
                'query': test_case['query'],
                'error': str(e),
                'iterations': 0,
                'is_adequate': False
            })
    
    # Summary report
    print("\n" + "=" * 50)
    print("📊 SUMMARY REPORT")
    print("=" * 50)
    
    successful_tests = [r for r in results if not r.get('error')]
    failed_tests = [r for r in results if r.get('error')]
    
    print(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed tests: {len(failed_tests)}")
    
    if successful_tests:
        avg_iterations = sum(r['iterations'] for r in successful_tests) / len(successful_tests)
        adequate_answers = sum(1 for r in successful_tests if r['is_adequate'])
        
        print(f"📈 Average iterations: {avg_iterations:.1f}")
        print(f"🎯 Adequate answers: {adequate_answers}/{len(successful_tests)}")
        
        # Show iteration distribution
        iteration_counts = {}
        for r in successful_tests:
            iterations = r['iterations']
            iteration_counts[iterations] = iteration_counts.get(iterations, 0) + 1
        
        print(f"📊 Iteration distribution:")
        for iterations, count in sorted(iteration_counts.items()):
            print(f"   {iterations} iteration(s): {count} test(s)")
    
    if failed_tests:
        print(f"\n❌ Failed test details:")
        for r in failed_tests:
            print(f"   - {r['test_name']}: {r['error']}")
    
    print(f"\n🔬 Full results saved to test_results.json")
    
    # Save detailed results
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

async def test_max_iterations():
    """Test that the system respects the maximum iteration limit"""
    print("\n🔄 Testing Maximum Iteration Limit")
    print("=" * 50)
    
    # Set a very low threshold to force iteration limit
    os.environ['MAX_ITERATIONS'] = '2'
    
    # Use a very vague query that's likely to be inadequate
    vague_query = "stuff about things"
    
    try:
        result = await process_rag_query(vague_query)
        metadata = result.get('metadata', {})
        iterations = metadata.get('iterations', 0)
        
        print(f"Query: '{vague_query}'")
        print(f"Iterations performed: {iterations}")
        print(f"Maximum allowed: 2")
        
        if iterations <= 2:
            print("✅ Maximum iteration limit respected")
        else:
            print("❌ Maximum iteration limit exceeded!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Reset to default
    os.environ['MAX_ITERATIONS'] = '3'

if __name__ == "__main__":
    try:
        asyncio.run(test_agentic_workflow())
        asyncio.run(test_max_iterations())
        print("\n🎉 All tests completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()
