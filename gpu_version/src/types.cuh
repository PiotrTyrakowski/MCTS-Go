#ifndef TYPES_H
#define TYPES_H

#include "constants.cuh"
#include "cuda_defs.hpp"

// If you still want a named max-size:
constexpr int MAX_SIZE = NN + 1;

struct IntPair {
    int first;
    int second;

    HOSTDEV constexpr IntPair(int first, int second) : first(first), second(second) {}
    HOSTDEV constexpr IntPair() : first(0), second(0) {}
};

struct ArrayInt {
    int array[MAX_SIZE];
    int _size;

    HOSTDEV ArrayInt() : _size(0) {}

    HOSTDEV ArrayInt(int _size) : _size(_size) {
        for(int i = 0; i < _size; ++i) {
            array[i] = 0;
        }
    }

    HOSTDEV ArrayInt(int _size, int fill) : _size(_size) {
        for(int i = 0; i < _size; ++i) {
            array[i] = fill;
        }
    }

    HOSTDEV int size() const {
        return _size;
    }

    HOSTDEV void push_back(int el) {
        array[_size++] = el;
    }

    HOSTDEV int remove_last() {
        return array[--_size];
    }

    HOSTDEV int& operator[](int index) {
        return array[index];
    }

    HOSTDEV const int& operator[](int index) const {
        return array[index];
    }
};

struct Array4Neighbors {
    int array[4];
    int _size;

    HOSTDEV constexpr Array4Neighbors() : array{ -1, -1, -1, -1 }, _size(0) {}


    HOSTDEV constexpr int size() const {
        return _size;
    }

    HOSTDEV constexpr void push_back(int el) {
        array[_size++] = el;
    }


    HOSTDEV constexpr int& operator[](int index) {
        return array[index];
    }

    HOSTDEV constexpr const int& operator[](int index) const {
        return array[index];
    }
};






struct ArrayIntPair {
    ArrayInt first;
    ArrayInt second;

    HOSTDEV ArrayIntPair() {}
    
    HOSTDEV ArrayIntPair(const ArrayInt& first, const ArrayInt& second)
        : first(first), second(second) {}
};

struct Queue {
    ArrayInt queue;
    int first_index;
    int last_index;

    HOSTDEV Queue() : first_index(0), last_index(-1) {}

    HOSTDEV bool empty() {
        return (first_index > last_index);
    }

    // add data to the end
    HOSTDEV void push(int el) {
        last_index++;
        queue[last_index] = el;
    }

    // remove first element and returns its value
    HOSTDEV int pop() {
        if (empty()) return -1; 
        return queue[first_index++];
    }
};

struct UnorderedSet {
    ArrayInt set;
    ArrayInt where;

    HOSTDEV UnorderedSet() : set(), where(MAX_SIZE, -1) {}

    HOSTDEV int size() {
        return set.size();
    }

    HOSTDEV void insert(int el) {
        // Bounds checking
        if(el < 0 || el >= MAX_SIZE) return;

        // element already exists
        if(where.array[el] >= 0) return;

        // Check if the set is full
        if(set.size() >= MAX_SIZE) return;

        where.array[el] = set.size();
        set.push_back(el);
    }

    HOSTDEV void remove(int el) {
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

    HOSTDEV bool contains(int el) const {
        if(el < 0 || el >= MAX_SIZE) return false;
        return where.array[el] >= 0;
    }

    HOSTDEV const ArrayInt& get() const {
        return set;
    }

    HOSTDEV ArrayInt& get() {
        return set;
    }
};

#endif // TYPES_H
