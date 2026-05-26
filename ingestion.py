import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates, for making tons of requests for tavily API - it might work only
# without using a VPN
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

load_dotenv()

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     show_progress_bar=False,
#     chunk_size=50, # it limits how many text objects langchain documents will send to openAI to embed them.
# # we have a token per request limitations that we can embed. (rate limits) it the number is too high - rate limit
#     retry_min_seconds=10, #
# )
embeddings = OllamaEmbeddings(model="nomic-embed-text", temperature=0.2, dimensions=1024)

# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index-cla", embedding=embeddings
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation from https://python.langchain.com",
        Colors.PURPLE,
    )

    # Crawl the documentation site
    # When we invoke the tavily crawler, hundreds of paths are explored in parallel with builtin extraction and
    # intelligent discovery
    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 1, #how far from the base url we want to go in the crawl. 1 means only the base url,
            # 2 means the base url and all the urls linked from it, etc. start with depths with 1 or 2, and only then
            # after reviewing the results wo can go deeper. it's going to be cheaper. 5 was chosen after some iterations.
            "extract_depth": "advanced", #advanced extraction means data tables, embedding content, with higher success
            # rate, but it might increase the latency
            "instructions": "content on ai agents" #very important, it helps tavily to know when to crawl a page and
            # when not, it's like a filter. with instructions, max_depth can be higher
        }
    )
    #
    # Convert Tavily crawl results to LangChain Document objects
    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]})
                for result in res["results"]]

    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site"
    )

    # for tavily_crawl_result_item in res["results"]:
    #     log_info(
    #         f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
    #     )
    #     all_docs.append(
    #         Document(
    #             page_content=tavily_crawl_result_item["raw_content"],
    #             metadata={"source": tavily_crawl_result_item["url"]},
    #         )
    #     )
    #
    # # Split documents into chunks
    # log_header("DOCUMENT CHUNKING PHASE")
    # log_info(
    #     f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
    #     Colors.YELLOW,
    # )
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    # splitted_docs = text_splitter.split_documents(all_docs)
    # log_success(
    #     f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    # )
    #
    # # Process documents asynchronously
    # await index_documents_async(splitted_docs, batch_size=500)
    #
    # log_header("PIPELINE COMPLETE")
    # log_success("🎉 Documentation ingestion pipeline finished successfully!")
    # log_info("📊 Summary:", Colors.BOLD)
    # log_info(f"   • Documents extracted: {len(all_docs)}")
    # log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    # we want to run the main coroutine using asyncio, since we have a lot of async calls to make to the vector store
    # and to the tavily API. We want to run them concurrently to speed up the process.
    asyncio.run(main())