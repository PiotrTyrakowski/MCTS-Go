#include "constants.hpp"

// Ensure constants.hpp defines NN appropriately for device usage.

const int MAX_SIZE = NN + 1;

struct IntPair {
    int first;
    int second;

    IntPair(int first, int second) : first(first), second(second) {}
    
    IntPair() : first(0), second(0) {}
};

struct ArrayInt {
    int array[MAX_SIZE];
    int _size;

    ArrayInt() : _size(0){
        
    }

    ArrayInt(int _size) : _size(_size) {
        for(int i = 0; i < _size; ++i) {
            array[i] = 0;
        }
    }

    ArrayInt(int _size, int fill) : _size(_size) {
        for(int i = 0; i < _size; ++i) {
            array[i] = fill;
        }
    }

    int size() const {
        return _size;
    }

    void push_back(int el) {
        array[_size++] = el;
    }

    int remove_last()
    {
        return array[--_size];
    }

    int& operator[](int index) {
        return array[index];
    }

    const int& operator[](int index) const {
        return array[index];
    }
};





struct ArrayIntPair {
    ArrayInt first;
    ArrayInt second;

    ArrayIntPair() {}
    
    ArrayIntPair(const ArrayInt& first, const ArrayInt& second)
        : first(first), second(second) {}
};

struct Queue {
    ArrayInt queue;

    int first_index;
    int last_index;

    Queue() : first_index(0), last_index(-1) {}

    bool empty() {
        return (first_index > last_index);
    }

    // add data to the end
    void push(int el) {
        last_index++;
        queue[last_index] = el;
    }

    // remove first element and returns its value
    int pop() {
        if (empty()) return -1; 

        return queue[first_index++];
    }

};

struct UnorderedSet {
    ArrayInt set;
    ArrayInt where;

    UnorderedSet() : set(), where(MAX_SIZE, -1) {}

    int size()
    {
        return set.size();
    }

    void insert(int el) {
            // Bounds checking
            if(el < 0 || el >= MAX_SIZE) return;

            // element already exists
            if(where.array[el] >= 0) return;

            // Check if the set is full
            if(set.size() >= MAX_SIZE) return;

            where.array[el] = set.size();
            set.push_back(el);
            
    }

    void remove(int el) {

         // Bounds checking
        if(el < 0 || el >= MAX_SIZE) return;

        int pos = where.array[el];

        // Check if the element exists in the set
        if(pos == -1) return;

        // Get the last element in the set
        int last_el = set.remove_last();

        // Move the last element to the position of the element to remove
        set.array[pos] = last_el;
        where.array[last_el] = pos;

        // Update 'where' for the removed element
        where.array[el] = -1;

    }

    bool contains(int el) const {
        if(el < 0 || el >= MAX_SIZE) return false;
        return where.array[el] >= 0;
    }

    const ArrayInt& get() const {
        return set;
    }

    ArrayInt& get() {
        return set;
    }



};
