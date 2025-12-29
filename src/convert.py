#!/usr/bin/env python3
"""Convert chat TXT files to JSON format."""

import json
import re
from pathlib import Path


def parse_chat_file(file_path: Path, min_answer_length: int = 100) -> list[dict]:
    """Parse a chat TXT file and return list of Q&A objects.

    Args:
        file_path: Path to the input TXT file
        min_answer_length: Minimum answer length in characters (default: 150)
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove metadata lines (starting with #)
    lines = content.split('\n')
    content_lines = [line for line in lines if not line.strip().startswith('#')]
    content = '\n'.join(content_lines)

    # Pattern to match speaker labels at the start of a line
    # Speaker label: A name/title followed by colon, at the start of line
    # Valid examples: "В.Путин:", "Читатель:", "Вопрос:", "А.Иванов:"
    # Must start with a capital Cyrillic/Latin letter
    # Must NOT be common sentence starters like "Так вот:", "Другой путь:", etc.
    # Limit length to avoid matching sentence fragments
    speaker_pattern = re.compile(
        r'^([А-ЯЁA-Z][А-Яа-яЁёA-Za-z\.\-]{0,30}(?:\s[А-Яа-яЁёA-Za-z\.\-]{1,20})?):\s*',
        re.MULTILINE
    )

    # Common false positives that look like speaker labels but are sentence starters
    false_speaker_patterns = {
        'Так вот', 'Другой путь', 'Первое', 'Второе', 'Третье',
        'Четвёртое', 'Пятое', 'Шестое', 'Седьмое', 'Далее', 'Кроме того', 'Например',
        'Во-первых', 'Во-вторых', 'В-третьих', 'Итак', 'Таким образом',
        'С одной стороны', 'С другой стороны', 'Иными словами',
        'Иначе говоря', 'То есть', 'Проще говоря', 'Короче говоря',
        'Я Вам скажу', 'Я Вам скажу и другое'
    }

    # Split content by speaker labels
    parts = speaker_pattern.split(content)

    # parts[0] is text before first speaker (usually empty or whitespace)
    # Then alternating: speaker, text, speaker, text, ...

    # Reconstruct, skipping false positives
    parsed = []
    current_speaker = None
    current_text_parts = []

    i = 1  # Start from index 1 (skip pre-first-speaker content)
    while i < len(parts) - 1:
        speaker = parts[i].strip()
        text = parts[i + 1]

        if speaker in false_speaker_patterns:
            # This is a false positive - append as text to current speaker
            current_text_parts.append(speaker + ':')
            current_text_parts.append(text)
        else:
            # Real speaker label
            # Save previous speaker's content if exists
            if current_speaker is not None:
                full_text = ' '.join(current_text_parts).strip()
                full_text = re.sub(r'\s+', ' ', full_text)
                if full_text:
                    parsed.append((current_speaker, full_text))

            current_speaker = speaker
            current_text_parts = [text]

        i += 2

    # Don't forget the last speaker
    if current_speaker is not None:
        full_text = ' '.join(current_text_parts).strip()
        full_text = re.sub(r'\s+', ' ', full_text)
        if full_text:
            parsed.append((current_speaker, full_text))

    # Now group into Q&A objects
    # Each В.Путин answer (or consecutive answers) creates one Q&A object
    # with all preceding questions since the last answer

    result = []
    current_questions = []
    current_questioners = []
    current_answers = []

    for speaker, text in parsed:
        if speaker == 'В.Путин':
            # This is an answer - add to current answers
            current_answers.append(text)
        else:
            # This is a question
            # If we have accumulated answers, save the previous Q&A and start fresh
            if current_answers:
                if current_questions:  # Only save if we have questions
                    answer_text = ' '.join(current_answers)
                    # Filter out short answers
                    if len(answer_text) >= min_answer_length:
                        result.append({
                            'source': file_path.name,
                            'questions': ' '.join(current_questions),
                            'answer': answer_text,
                            'questioners': list(dict.fromkeys(current_questioners))
                        })
                current_questions = []
                current_questioners = []
                current_answers = []

            # Add this question
            current_questions.append(text)
            if speaker not in current_questioners:
                current_questioners.append(speaker)

    # Handle any remaining data (last Q&A)
    if current_questions and current_answers:
        answer_text = ' '.join(current_answers)
        # Filter out short answers
        if len(answer_text) >= min_answer_length:
            result.append({
                'source': file_path.name,
                'questions': ' '.join(current_questions),
                'answer': answer_text,
                'questioners': list(dict.fromkeys(current_questioners))
            })

    return result


def convert_file(input_path: Path, output_path: Path):
    """Convert a single TXT file to JSON."""
    data = parse_chat_file(input_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Converted {input_path.name} -> {output_path.name} ({len(data)} Q&A pairs)")


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    original_dir = script_dir / '../data/chats'

    # Convert all .txt files in the directory
    txt_files = sorted(original_dir.glob('*.txt'))

    for txt_file in txt_files:
        json_file = txt_file.with_suffix('.json')
        convert_file(txt_file, json_file)

    print(f"\nTotal: {len(txt_files)} files converted")


if __name__ == '__main__':
    main()
