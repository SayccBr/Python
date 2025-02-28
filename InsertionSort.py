def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        # Move os elementos do arr[0..i-1], que são maiores que a chave,
        # para uma posição à frente de sua posição atual
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def print_array(arr):
    for i in range(len(arr)):
        print(arr[i], end=" ")
    print()

# Exemplo de uso
arr = [12, 11, 13, 5, 6]
print("Array original:")
print_array(arr)

insertion_sort(arr)

print("Array ordenado:")
print_array(arr)