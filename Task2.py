class Node:
    """A node in a singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None
class LinkedList:
    """Singly linked list with basic operations."""
    def __init__(self):
        self.head = None
    def add_node(self, data):
        """Add a node with the given data to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = new_node
    def print_list(self):
        """Print all elements in the linked list."""
        if not self.head:
            print("The list is empty.")
            return
        curr = self.head
        while curr:
            print(curr.data, end=" -> ")
            curr = curr.next
        print("None")
    def delete_nth_node(self, n):
        """Delete the nth node (1-based index) from the list."""
        if not self.head:
            raise Exception("Cannot delete from an empty list.")
        if n <= 0:
            raise ValueError("Index must be 1 or greater.")
        if n == 1:
            print(f"Deleting node {n} (value: {self.head.data})")
            self.head = self.head.next
            return
        curr = self.head
        for _ in range(n - 2):
            if curr.next is None:
                raise IndexError("Index out of range.")
            curr = curr.next
        if not curr.next:
            raise IndexError("Index out of range.")
        print(f"Deleting node {n} (value: {curr.next.data})")
        curr.next = curr.next.next

if __name__ == "__main__":
    ll = LinkedList()
    print("Adding nodes 12, 24, 36, 48, 50 to the list.")
    ll.add_node(12)
    ll.add_node(24)
    ll.add_node(36)
    ll.add_node(48)
    ll.add_node(50)
    
    print("List:")
    ll.print_list()

    try:
        print("\nDeleting 3rd node:")
        ll.delete_nth_node(3)
        ll.print_list()

        print("\nTrying to delete 10th node :")
        ll.delete_nth_node(12)
    except Exception as e:
        print("Exception:", e)

    try:
        print("\nCreating empty list and trying to delete:")
        empty_list = LinkedList()
        empty_list.delete_nth_node(1)
    except Exception as e:
        print("Exception:", e)
