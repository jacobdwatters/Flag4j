#
#   MIT License
#
#   Copyright (c) 2025. Jacob Watters
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#

import glob
import re


def process_javadoc(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        print(f"[LATEX CONVERT] -- parsing file {file_path}")

    # Regex pattern to find <span> blocks followed immediately by a <!-- LATEX: block
    pattern = re.compile(
        r'(<span class="latex-replaceable">.*?</span>)\s*<!-- LATEX:\s*({@literal\s*.*?\s*}\s*) -->',
        re.DOTALL
    )

    # Function to replace matched spans with their corresponding LaTeX block
    def replace_match(match):
        span_block = match.group(1)  # The <span> block
        latex_block = match.group(2)  # The LaTeX content inside the comment
        return f"\n{latex_block}\n"  # Replace span with LaTeX block

    # Apply the replacement
    updated_content = pattern.sub(replace_match, content)

    # Write changes back to the file if any replacements were made
    if updated_content != content:
        print(f"[LATEX CONVERT] --       making LaTeX replacement")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated_content)

# Process all Javadoc HTML files in
for file in glob.glob("target/reports/apidocs/**/*.html", recursive=True):
    process_javadoc(file)

print("[LATEX CONVERT] -- complete")
