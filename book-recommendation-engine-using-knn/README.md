# Book Recommendation Engine using KNN

freeCodeCamp Machine Learning with Python project solution for the Book Recommendation Engine using KNN.

The implementation builds a user-rating vector for each frequently rated book in the Book-Crossings dataset, filters sparse users/books with the FCC thresholds, and uses `sklearn.neighbors.NearestNeighbors` with cosine distance to return the five closest books.

## Files

- `book_recommender.py` — solution module with the required `get_recommends(book)` function.
- `main.py` — simple demo entry point.
- `test_book_recommender.py` — local smoke tests for FCC return shape and sorted nearest-neighbor distances.
- `.gitignore` — excludes downloaded data, caches, and virtualenvs.

## Verification

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python main.py
```

On first run, `book_recommender.py` downloads `https://cdn.freecodecamp.org/project-data/books/book-crossings.zip` into `data/`. The dataset is intentionally ignored by git.
