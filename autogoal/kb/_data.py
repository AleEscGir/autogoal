__all__ = [
    "DataType",
    "Document",
    "Sentence",
    "Word",
    "Category",
    "Vector",
    "Matrix",
    "DenseMatrix",
    "SparseMatrix",
    "List",
]


class DataType:
    def __init__(self, **tags):
        self.tags = tags

    def get_tag(self, tag):
        return self.tags.get(tag, None)

class Word(DataType):
    pass

class Sentence(DataType):
    pass

class Document(DataType):
    pass

class Category(DataType):
    pass

class Vector(DataType):
    pass

class Matrix(DataType):
    pass

class DenseMatrix(DataType):
    pass

class SparseMatrix(DataType):
    pass

class List(DataType):
    def __init__(self, inner_type):
        self._inner_type = inner_type
        super().__init__(**inner_type.tags)


# class Classifier:
#     def run(self, input: Document(domain='health', language='english')) -> Category():
#         pass
