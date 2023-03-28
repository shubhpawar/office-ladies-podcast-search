import os
import re


NO_TIMESTAMP_FILES = ['episode1', 'episode2', 'episode3']

# Read a file from the data/transcripts directory which is one level up from the current directory and return a list of lines
def read_file_lines(filename):
    with open(os.path.join('..', 'data', 'transcripts', filename), 'r') as f:
        return f.readlines()


def extract_episode_info(lines):
    # Extract the episode number and title from the 13th line of the transcript
    lines[12] = lines[12].replace(' – ', ' - ')
    episode_info = lines[12].split(' - ')
    episode_number = episode_info[0].split(' | ')[-1].strip()
    episode_title = episode_info[1].strip()

    return episode_number, episode_title


def extract_line_metadata(line):
    # Extract the speaker and timestamp from the line
    # Check if the timestamp is present like [00:00:00]
    if not re.search(r'\[\d{2}:\d{2}:\d{2}\]', line):
        line_metadata = line.split(':')
        speaker = line_metadata[0].strip()
        timestamp = None
        text = line_metadata[1].strip()
    else:
        # Check if line starts with a timestamp like [timestamp] speaker text
        if re.search(r'^\[\d{2}:\d{2}:\d{2}\]', line):
            line_metadata = line.split(']')
            timestamp = line_metadata[0].strip().replace('[', '')
            # First word of the line is the speaker
            speaker = line_metadata[1].strip().split(' ')[0]
            # Rest of the line is the text
            text = ' '.join(line_metadata[1].strip().split(' ')[1:])
        else:
            # Example line: speaker [timestamp] text
            line_metadata = line.split('[')
            speaker = line_metadata[0].strip()
            timestamp = line_metadata[1].split(']')[0].strip()
            text = line_metadata[1].split(']')[1].strip()

    return speaker, timestamp, text


def clean_and_extract_data(filename):
    lines = read_file_lines(filename)

    # Extract the episode number and title 
    episode_number, episode_title = extract_episode_info(lines)

    # Remove the first 12 lines which are not part of the transcript
    lines = lines[12:]

    # Remove the new line character from each line and remove empty lines
    lines = [line.strip() for line in lines if line.strip()]

    # Remove lines that are "COMMERCIAL BREAK" or "BREAK" or "-"
    lines = [line for line in lines if line != 'COMMERCIAL BREAK' and line != 'BREAK' and line != '-']
    
    # Find the index of the first line that contains the phrase "We're the Office Ladies" as a substring
    for i, line in enumerate(lines):
        if "we're the office ladies" in line.replace('"', '').lower():
            index = i
            break

    # Remove all lines before and including the line that contains the phrase "We're the Office Ladies"
    lines = lines[index + 1:]

    # Find the index of the last line that contains the phrase "Thank you for listening to Office Ladies" as a substring and remove all lines after that line (including the line that contains the phrase)
    for i, line in enumerate(lines):
        if "thank you for listening to office ladies" in line.replace('"', '').lower():
            index = i

    lines = lines[:index]

    # Extract the speaker, timestamp, and text from each line and store the data in a list of disctionaries
    lines_data = []
    for line in lines:
        speaker, timestamp, text = extract_line_metadata(line)
        lines_data.append({
            'speaker': speaker,
            'timestamp': timestamp,
            'text': text
        })

    # Create a dictionary with the episode number, title, and lines data
    episode_data = {
        'episode_number': episode_number,
        'episode_title': episode_title,
        'lines': lines_data
    }

    return episode_data