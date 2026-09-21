from chunking.chunking import find_best_boundary


text = "The patient has diabetes and needs regular monitoring of blood glucose levels"

position = find_best_boundary(text, 0, 35)

print("Boundary position:", position)
print("Text before boundary:")
print(text[:position])