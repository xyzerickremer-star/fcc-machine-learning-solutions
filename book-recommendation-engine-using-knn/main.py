"""Small entry point mirroring freeCodeCamp's project notebook usage."""

from pprint import pprint

from book_recommender import get_recommends


if __name__ == "__main__":
    pprint(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
