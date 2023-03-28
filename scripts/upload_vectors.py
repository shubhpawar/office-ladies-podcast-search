import os
import tqdm
import argparse

import pinecone

from clean_data import clean_and_extract_data
from embed_data import create_segments, embed_segments


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
METADATA_FIELDS = ['episode_title', 'episode_number', 'start_time', 'end_time', 'text']

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="us-west4-gcp"
)

def prepare_batch(transcript_segments, metadata_fields, batch_size):
    for i in range(0, len(transcript_segments), batch_size):
        i_end = min(len(transcript_segments) - 1, i + batch_size)

        # Extract the ids from the transcript segments
        ids = [segment['id'] for segment in transcript_segments[i:i_end+1]]

        # Extract the vectors from the transcript segments
        vectors = [segment['vector'] for segment in transcript_segments[i:i_end+1]]

        # Extract the metadata from the transcript segments if the value is not None
        metadata = [{field: segment[field] for field in metadata_fields if segment[field] is not None} for segment in transcript_segments[i:i_end+1]]

        yield ids, vectors, metadata


def upsert_vectors(transcript_segments, index_name, metadata_fields=[], batch_size=100):
    index = pinecone.Index(index_name)

    for ids, vectors, metadata in prepare_batch(transcript_segments, metadata_fields, batch_size):
        batch_data = list(zip(ids, vectors, metadata))
        index.upsert(batch_data)
        print(f"Upserted {len(batch_data)} vectors")


if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()

    # Episode number or "all" to process all episodes
    parser.add_argument('--episode', type=str, default="all", help='Episode filename or "all" to process all episodes')

    # Pinecone index name
    parser.add_argument('--index', type=str, default="shubhams-index", help='Pinecone index name')

    args = parser.parse_args()

    if args.episode == "all":
        # Get all file names in the data directory
        episodes = os.listdir("../data/transcripts")

        # Remove files that are not episodes
        episodes = [episode for episode in episodes if episode.startswith("episode")]

        # Sort the episodes by episode number
        episodes = sorted(episodes, key=lambda episode: int(episode.replace("episode", "")))
    else:
        episodes = [args.episode]

    print(f"Processing {len(episodes)} episodes: {episodes}")

    for episode in episodes:
        print(f"Processing {episode}")

        # Clean and extract the data
        print("Cleaning and extracting data")
        episode_data = clean_and_extract_data(episode)

        # Create the transcript segments
        print("Creating transcript segments")
        transcript_segments = create_segments(episode_data)

        # Embed the transcript segments
        print("Embedding transcript segments")
        transcript_segments = embed_segments(transcript_segments)

        # Upsert the transcript segments into Pinecone
        print("Upserting transcript segment vectors into Pinecone")
        upsert_vectors(transcript_segments, args.index, metadata_fields=METADATA_FIELDS)

    print("Done")