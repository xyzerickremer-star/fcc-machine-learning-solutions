"""freeCodeCamp Book Recommendation Engine using KNN.

This module implements the FCC certification project's required
``get_recommends(book)`` function using the Book-Crossings ratings dataset,
cosine distance, and a brute-force nearest-neighbor index over the
book-by-user ratings matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, cast
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

DATA_URL = "https://cdn.freecodecamp.org/project-data/books/book-crossings.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"
ZIP_PATH = DATA_DIR / "book-crossings.zip"
BOOKS_CSV = DATA_DIR / "BX-Books.csv"
RATINGS_CSV = DATA_DIR / "BX-Book-Ratings.csv"

MIN_USER_RATINGS = 200
MIN_BOOK_RATINGS = 100

# Lazily initialized globals.  FCC calls get_recommends directly, so delaying
# dataset download/model construction keeps imports cheap and test-friendly.
_BOOK_USER_MATRIX: pd.DataFrame | None = None
_MODEL: NearestNeighbors | None = None

Recommendation = Tuple[str, float]
FCCReturn = List[object]


def ensure_data() -> None:
    """Download and extract the Book-Crossings CSV files if they are absent."""
    if BOOKS_CSV.exists() and RATINGS_CSV.exists():
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading {DATA_URL} ...")
        request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response, ZIP_PATH.open("wb") as output:
            output.write(response.read())

    with ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)


def build_book_user_matrix() -> pd.DataFrame:
    """Load, filter, and pivot ratings into a book-title x user matrix."""
    ensure_data()

    books = pd.read_csv(
        BOOKS_CSV,
        sep=";",
        encoding="ISO-8859-1",
        usecols=["ISBN", "Book-Title"],
        dtype={"ISBN": "string", "Book-Title": "string"},
    )
    ratings = pd.read_csv(
        RATINGS_CSV,
        sep=";",
        encoding="ISO-8859-1",
        usecols=["User-ID", "ISBN", "Book-Rating"],
        dtype={"User-ID": "int32", "ISBN": "string", "Book-Rating": "int8"},
    )

    # FCC's intended de-noising: keep active users and frequently rated books.
    active_users = ratings["User-ID"].value_counts()
    active_users = active_users[active_users >= MIN_USER_RATINGS].index
    popular_books = ratings["ISBN"].value_counts()
    popular_books = popular_books[popular_books >= MIN_BOOK_RATINGS].index

    filtered = ratings[
        ratings["User-ID"].isin(active_users) & ratings["ISBN"].isin(popular_books)
    ]
    merged = filtered.merge(books, on="ISBN")

    matrix = merged.pivot_table(
        index="Book-Title",
        columns="User-ID",
        values="Book-Rating",
        aggfunc="mean",
        fill_value=0,
    )
    return matrix.sort_index()


def get_model() -> tuple[pd.DataFrame, NearestNeighbors]:
    """Return the cached ratings matrix and fitted nearest-neighbor model."""
    global _BOOK_USER_MATRIX, _MODEL
    if _BOOK_USER_MATRIX is None or _MODEL is None:
        matrix = build_book_user_matrix()
        sparse_matrix = csr_matrix(matrix.values)
        model = NearestNeighbors(metric="cosine", algorithm="brute")
        model.fit(sparse_matrix)
        _BOOK_USER_MATRIX = matrix
        _MODEL = model
    return cast(pd.DataFrame, _BOOK_USER_MATRIX), cast(NearestNeighbors, _MODEL)


def get_recommends(book: str = "") -> FCCReturn:
    """Return the five nearest book recommendations for ``book``.

    The return shape matches freeCodeCamp's expected API:
    ``[query_book, [[recommended_title, cosine_distance], ...]]``.
    Recommendations are ordered from closest to farthest and exclude the query
    book itself.
    """
    matrix, model = get_model()
    if book not in matrix.index:
        raise ValueError(f"Book not found after FCC filtering: {book!r}")

    query_row = matrix.index.get_loc(book)
    distances, indices = model.kneighbors(
        [matrix.iloc[query_row].to_numpy()], n_neighbors=6
    )

    recommendations: list[Recommendation] = []
    for distance, index in zip(distances[0], indices[0]):
        title = str(matrix.index[index])
        if title == book:
            continue
        recommendations.append((title, float(distance)))

    return [book, recommendations[:5]]


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
