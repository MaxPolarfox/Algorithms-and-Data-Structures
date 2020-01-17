// TYPICAL APPROACHES to solve problems:

// FREQUENCY COUNTERf
const validAnagram = (str1, str2) => {
  let fr1 = {};

  if (str1.length !== str2.length) return false
  for (let char of str1) {
    fr1[char] = (fr1[char] || 0) + 1
  }

  for (let char of str2) {
    if (!(fr1[char])) return false;
    else fr1[char] -= 1;
  }
  return true
}


// MULTIPLE POINTERS
const coutUnqueValues = (arr) => {
  if (!arr.length) return 0
  let i = 0;
  for (let j = 0; j < arr.length; j++)
    if (arr[i] !== arr[j]) {
      i++;
      arr[i] = arr[j]
    }
  return i + 1;
}


// IndexOf implementstion
const indexOf = (target, str) => {
  let targetPointer = 0;
  let strPointer = 0
  while (strPointer <= str.length - target.length) {
    if (str[strPointer] === target[targetPointer]) {
      targetPointer++;
    } else {
      targetPointer = 0;
    }
    if (targetPointer === target.length) return strPointer + 1 - target.length;

    strPointer++;
  }
  return -1;
}

// Average Pair
function averagePair(arr, num) {
  let start = 0;
  let end = arr.length - 1;
  while (start < end) {
    let avr = (arr[start] + arr[end]) / 2;
    if (avr === num) return true;
    else if (avr < num) start++;
    else end--;
  }
  return false;
}

// isSubsequense
function isSubsequence(str1, str2) {
  let i = 0;
  let j = 0;
  while (j < str2.length) {
    if (str1[i] === str2[j]) i++;
    else i = 0;
    if (i === str1.length) return true
    j++;
  }
  return false
}


function maxSubarraySum(arr, num) {
  if (arr.length < num) return null;
  let maxSum = 0;
  for (let i = 0; i < num; i++) {
    maxSum += arr[i];
  }
  let tempSum = maxSum;
  for (let i = num; i < arr.length; i++) {
    tempSum += arr[i] - arr[i - num];
    maxSum = Math.max(tempSum, maxSum)
  }
  return maxSum
}


function minSubArrayLen(arr, sum) {
  let total = 0;
  let start = 0;
  let end = 0;
  let minLen = Infinity;
  while (start < arr.length) {
    if (total < sum && end < arr.length) {
      total += arr[end];
      end++;
    } else if (total >= sum) {
      minLen = Math.min(minLen, end - start)
      total -= arr[start]
      start++;
    } else {
      break;
    }
  }
  return minLen === Infinity ? 0 : minLen;
}

function findLongestSubstring(str) {
  let start = 0;
  let longest = 0;

  let obj = {};

  for (let i = 0; i < str.length; i++) {
    let char = str[i]

    if (obj[char]) {
      start = Math.max(start, obj[char])
    }

    longest = Math.max(longest, i - start + 1)

    obj[char] = i + 1;
  }
  return longest
}

function reverse(str) {
  let last = str[str.length - 1]
  if (!last) return '';
  else return last + reverse(str.slice(0, str.length - 1))
}


function binarySearch(arr, target) {
  let start = 0;
  let end = arr.length - 1;
  let middle = Math.floor((start + end) / 2);

  while (arr[middle] !== target && start <= end) {
    if (target < arr[middle]) end = middle - 1;
    else start = middle + 1;
    middle = Math.floor((end + start) / 2);
  }
  return arr[middle] === target ? middle : -1;
}


// find how many times target shows up in str;
function naiveSearch(target, str) {
  let counter = 0;
  for (let i = 0; i < str.length - target.length; i++) {
    for (let j = 0; j < target.length; j++) {
      if (str[i + j] !== target[j]) break;
      if (target.length - 1 === j) counter++;
    }
  }
  return counter;
}

// END OF TYPICAL APPROACHES to solve problems;



// SORTING ALGORITHMS:

/*
BUBBLE SORT:
Time: O(N^2)
Space: O(1)
*/
function bubbleSort(arr) {
  let temp;
  for (let i = 0; i < arr.length; i++) {
    for (let j = 0; j < arr.length - i; j++) {

      if (arr[j] > arr[j + 1]) {
        temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp
      }
    }
  }
  return arr
}

const bubbleSortOptimized = (arr) => {
  let noSwaps
  let temp;
  for (let i = arr.length; i > 0; i--) {
    noSwaps = true
    for (let j = 0; j < i - 1; j++) {
      if (arr[j] > arr[j + 1]) {
        temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp
        noSwaps = false;
      }
    }
    if (noSwaps) break;
  }
  return arr
}


/* SELECTION SORT
Time: O(N^2)
Space: O(1)
*/
const selectionSort = (arr) => {
  for (let i = 0; i < arr.length; i++) {
    let smallest = i
    for (let j = i + 1; j < arr.length; j++) {

      if (arr[smallest] > arr[j]) smallest = j;

      if (j !== smallest) {
        let small = arr[smallest];
        arr[smallest] = arr[i];
        arr[i] = small;
      }
    }
  }
  return arr
}



/*INSERTION SORT:
  Time: O(N^2)
  Space: O(1)
*/
const insertionSort = (arr) => {
  for (let i = 1; i < arr.length; i++) {
    let currentVal = arr[i];

    for (let j = i - 1; j >= 0; j--) {
      if (arr[j] > currentVal) {
        arr[j + 1] = arr[j];
        arr[j] = currentVal;
      } else {
        break
      }
    }
  }
  return arr
}


/*
MERGE SORT
Time: O(N * Log N)
Space: O(N)
*/
const merge = (arr1 = [], arr2 = []) => {
  let result = []
  let i = 0;
  let j = 0;

  while (i < arr1.length && j < arr2.length) {
    if (arr1[i] < arr2[j]) {
      result.push(arr1[i])
      i++
    } else {
      result.push(arr2[j])
      j++
    }
  }

  while (i < arr1.length) {
    result.push(arr1[i])
    i++;
  }
  while (j < arr2.length) {
    result.push(arr2[j])
    j++;
  }
  return result
}

const mergeSort = (arr) => {
  if (arr.length <= 1) return arr;

  let middle = Math.floor(arr.length / 2);

  let left = mergeSort(arr.slice(0, middle));
  let right = mergeSort(arr.slice(middle))

  return merge(left, right)
}


/*
QUICK SORT
Time: O(N * log N) || O(N^2)
Space: O(log N)
*/
function pivot(arr, start = 0, end = arr.length - 1) {
  const swap = (arr, idx1, idx2) => {
    [arr[idx1], arr[idx2]] = [arr[idx2], arr[idx1]];
  };

  // We are assuming the pivot is always the first element
  let pivot = arr[start];
  let swapIdx = start;

  for (let i = start + 1; i <= end; i++) {
    if (pivot > arr[i]) {
      swapIdx++;
      swap(arr, swapIdx, i);
    }
  }

  // Swap the pivot from the start the swapPoint
  swap(arr, start, swapIdx);
  return swapIdx;
}

const quickSort = (arr, left = 0, right = arr.length - 1) => {
  if (left < right) {
    let pivotInd = pivot(arr, left, right);

    quickSort(arr, left, pivotInd - 1);
    quickSort(arr, pivotInd + 1, right)
  }
  return arr
}


/*
RADIX SORT
Time: O (NK)
Space: (N + K)
*/
const getDigit = (num, index) => Math.floor(Math.abs(num) / Math.pow(10, index) % 10)

const digitCount = (num) => {
  if (num === 0) return 1;
  return Math.floor(Math.log10(Math.abs(num)) + 1)
}

const mostDigit = arr => {
  let most = 0;
  for (let dig of arr) {
    let current = digitCount(dig)
    most = Math.max(current, most)
  }
  return most
}

const radixSort = (arr) => {
  let most = mostDigit(arr);

  for (let i = 0; i < most; i++) {
    let buckets = Array.from({ length: 10 }, () => [])
    arr.forEach(num => buckets[getDigit(num, i)].push(num))
    arr = [].concat(...buckets)
  }
  return arr
}


/*
implement Single LL:
insertion: O(1)
removal O(1) || O(N)
search O(N)
access O(N)
*/
class Node {
  constructor(value, next = null) {
    this.value = value,
      this.next = next
  }
}

class SingleLinkedList {
  constructor() {
    this.head = null,
      this.tail = null,
      this.length = 0
  }

  push(val) {
    let newNode = new Node(val);

    if (!this.head) {
      this.head = newNode;
      this.tail = this.head;
    } else {
      this.tail.next = newNode;
      this.tail = newNode
    }

    this.length++;
  }

  pop() {
    if (!this.length) return undefined;
    let current = this.head;
    let newTail = current;

    while (current.next) {
      newTail = current;
      current = current.next;
    }
    this.tail = newTail;
    this.tail.next = null;
    this.length--;

    if (!this.length) {
      this.head = null;
      this.tail = null;
    }
    return current
  }

  shift() {
    if (this.head) {
      let currentHead = this.head;
      this.head = currentHead.next;
      this.length--;
      if (!this.length) {
        this.tail = null;
      }
      return currentHead
    } else {
      return undefined
    }
  }

  unshift(val) {
    let newNode = new Node(val);

    if (!this.length) {
      this.head = newNode;
      this.tail = this.head;
    } else {
      newNode.next = this.head;
      this.head = newNode;
    }
    this.length++
    return this
  }

  get(ind) {

    if (ind < 0 || ind >= this.length) return undefined;
    else {
      let currentInd = 0;
      let currentNode = this.head;

      while (currentInd !== ind) {
        currentNode = currentNode.next;
        currentInd++;
      }
      return currentNode
    }
  }

  set(ind, val) {
    let target = this.get(ind);

    if (!target) return false;
    else {
      target.value = val;
      return true
    }
  }

  insert(ind, val) {

    if (ind < 0 || ind > this.length) return false;

    if (ind === 0) return !!this.unshift(val);
    if (ind === this.length - 1) !!this.push(val);

    let beforeInsert = this.get(ind - 1);

    let newNode = newNode(val);
    newNode.next = beforeInsert.next;
    beforeInsert.next = newNode

    this.length++;
    return true
  }

  remove(ind) {
    if (ind < 0 || ind > this.length) return undefined;

    if (!ind) return this.shift();
    if (this.length - 1 === ind) return this.pop();

    let beforeRemove = this.get(ind - 1);
    let removed = beforeRemove.next;
    beforeRemove.next = removed.next;
    this.length--;

    return removed
  }

  reverse() {
    let node = this.head;

    this.head = this.tail;
    this.tail = node;

    let next;
    let prev = null;


    while (node.next) {

      next = current.next;
      node.next = prev;

      prev = current.val
      current = next;
    }
  }

  rotate(num) {
    num = num % this.length;

    let curr = this.head;

    this.tail.next = this.head;

    num = num > 0 ? num : this.length + num

    for (let i = 1; i < num; i++) {
      curr = curr.next;
    }

    this.head = curr.next;
    curr.next = null;
    this.tail = curr
  }
}

/*
  Doubly LL
insertion: O(1)
removal O(1) || O(N)
search O(N)
access O(N)
  */

class NodeDll {
  constructor(val, next = null, prev = null) {
    this.val = val;
    this.next = next;
    this.prev = prev;
  }
}

class DoublyLL {
  constructor() {
    this.head = null;
    this.next = null;
    this.tail = null;
    this.length = 0;
  }

  push(val) {
    let newNode = new NodeDll(val);

    if (!this.head) {
      this.head = newNode;
      this.tail = newNode;
    } else {
      this.tail.next = newNode;
      newNode.prev = this.tail;
      this.tail = newNode;
    }
    this.length++;
    return this
  }

  pop() {
    let popped = this.tail;
    if (!this.length) return undefined;
    if (this.length === 1) {
      this.head = null;
      this.tail = null;
    } else {
      this.tail = popped.prev
      this.tail.next = null
      popped.prev = null
    }
    this.length--;
    return popped
  }

  shift() {
    if (!this.head) return undefined;

    let shifted = this.head;

    if (this.length === 1) {
      this.head = null;
      this.tail = null;
    }
    shifted.next.prev = null;
    this.head = shifted.next
    shifted.next = null;

    this.length--;

    return shifted
  }

  unshift(val) {

    let newNode = new NodeDll(val);
    if (!this.length) {
      this.head === newNode;
      this.tail === newNode;
    }

    newNode.next = this.head;
    this.head.prev = newNode;
    this.head = newNode;

    this.length++;
    return this
  }

  get(ind) {
    if (ind < 0 || ind > this.length) return null

    let midPoint = Math.floor(this.length / 2);
    if (ind <= midPoint) {
      let current = this.head;
      for (let i = 0; i <= midPoint; i++) {
        if (ind === i) {
          let result = current;
          return result
        } else {
          current = current.next;
        }
      }
    } else {
      let current = this.tail;
      for (let i = this.length - 1; i > midPoint; i--) {
        if (ind === i) {
          let result = current;
          return result
        } else {
          current = current.prev;
        }
      }
    }
  }

  set(ind, val) {
    let target = this.get(ind);

    if (target !== null) {
      target.val = val;
      return true
    } else {
      return false
    }
  }

  insert(ind, val) {
    if (ind < 0 || ind > this.length) return null

    if (!ind) {
      this.unshift(val)
    } else if (ind === this.length) {
      this.push(val)
    } else {
      let newNode = new NodeDll(val);
      let after = this.get(ind);
      let before = after.prev;

      newNode.next = after;
      after.prev = newNode;

      before.next = newNode;
      newNode.prev = before
      this.length++
    }
    return true
  }

  remove(ind) {
    if (ind < 0 || ind > this.length) return null;

    if (!ind) {
      return this.pop()
    } else if (ind === this.length - 1) {
      return this.pop()
    } else {
      let removed = this.get(ind);
      let before = removed.prev;
      let after = removed.next;

      before.next = after;
      after.prev = before;

      removed.next = null;
      removed.prev = null;

      this.length--;
      return removed
    }
  }

  reverse() {
    let node = this.head;
    this.head = this.tail;
    this.tail = node;

    let prev = null;
    let next;

    for (let i = 0; i < this.length; i++) {
      next = node.next
      node.prev = next;
      node.next = prev;
      prev = node;
      node = next
    }
    return this
  }

}

/*
implement STACK
insertion: O(1)
removal O(1)
search O(N)
access O(N)
*/

class NodeStackQueue {
  constructor(value) {
    this.value = value;
    this.next = null;
  }
}

class Stack {
  constructor() {
    this.first = null;
    this.last = null;
    this.size = 0;
  }
  push(val) {
    let newNode = new NodeStackQueue(val);

    if (!this.first) {
      this.first = newNode;
      this.last = this.first;
    } else {
      let oldFirst = this.first;
      this.first = newNode;
      this.first.next = oldFirst;
    }
    this.size++;
  }
  pop() {
    if (!this.first) return null;

    let removed = this.first;

    if (this.size === 1) {
      this.first = null;
      this.last = null;
    } else {
      this.first = removed.next;
      removed.next = null
    }
    this.size--;
    return removed
  }
}

/*
Implement  QUEUE:
insertion: O(1)
removal O(1)
search O(N)
access O(N)
*/
class Queue {
  constructor() {
    this.first = null;
    this.last = null;
    this.size = 0;
  }
  enqueue(val) {
    let newNode = new NodeStackQueue(val);

    if (!this.size) {
      this.first = newNode;
      this.last = newNode;
    } else {
      this.last.next = newNode;
      this.last = newNode;
    }
    this.size++
  }

  dequeue() {
    if (!this.size) return null;

    let removed = this.first;

    if (this.size === 1) {
      this.first = null;
      this.last = null;
    } else {
      this.first = removed.next;
      removed.next = null
    }
    this.size--;
    return removed
  }
}


/*
Implement Queue with 2 Stacks
*/
class QueueWithTwoStacks {
  constructor() {
    this.s1 = new Stack();
    this.s2 = new Stack();
  }
  enqueue(val) {
    this.s1.push(val)
    return this.s1
  }
  dequeue() {
    if (this.s2.size === 0) {
      if (this.s1.size === 0) return null
      while (this.s1.size > 0) {
        var p = this.s1.pop();
        this.s2.push(p);
      }
    }
    return this.s2.pop();
  }
}


/*
Implement Stack with 2 Queues
*/

class Stack {
  constructor() {
    this.q1 = new Queue();
    this.q2 = new Queue();
  }
  push(val) {
    this.q1.enqueue(val)
    return this.q1
  }
  pop() {
    while (this.q1.size > 1) {
      let dequeued = this.q1.dequeue();
      this.q2.enqueue(dequeued);
    }
    let result = this.q1.dequeue();

    let temp = this.q1;
    this.q1 = this.q2;
    this.q2 = temp

    return result
  }
}

/*
BINARY SEARCH TREE:
insertion O(log N)
search O(log N)
*/

class NodeBST {
  constructor(value) {
    this.value = value;
    this.left = null;
    this.right = null;
  }
}

class BinarySearchTree {
  constructor() {
    this.root = null;
  }

  insert(value, current = this.root) {
    let newNode = new NodeBST(value);

    if (!this.root) {
      this.root = newNode;
      return this
    }

    if (current.value > value) {
      if (current.left) {
        current = current.left
        return this.insert(value, current)
      } else {
        current.left = newNode;
        return this

      }
    } else {
      if (current.right) {
        current = current.right
        return this.insert(value, current)
      } else {
        current.right = newNode;
        return this
      }
    }

  }


  find(val, current = this.root) {
    if (!this.root) return undefined;

    if (current.value === val) return current

    if (current.value > val) {
      if (current.left) {
        current = current.left
        return this.find(val, current)
      } else {
        return undefined

      }
    } else {
      if (current.right) {
        current = current.right
        return this.find(val, current)
      } else {
        return undefined
      }
    }
  }

  remove(value) {
    const removeNode = (node, value) => {
      if (!node) {
        return null;
      }

      if (value == node.value) {
        if (!node.left && !node.right) {
          return null;
        }

        if (!node.left) {
          return node.right;
        }

        if (!node.right) {
          return node.left;
        }

        let temp = node.right;

        while (!temp.left) {
          temp = temp.left;
        }

        node.value = temp.value;

        node.right = removeNode(node.right, temp.value);

      } else if (value < node.value) {
        node.left = removeNode(node.left, value);
        return node;

      } else {
        node.right = removeNode(node.right, value);
        return node;
      }
    }
    this.root = removeNode(this.root, value)
  }

  BreadthFirstSearch() {
    let queue = [this.root];
    let result = []

    while (queue.length) {
      let dequeued = queue.shift();
      result.push(dequeued.value);
      if (dequeued.left) queue.push(dequeued.left);
      if (dequeued.right) queue.push(dequeued.right);
    }
    return result
  }

  DFSPreoder() {
    let result = [];

    const traverse = (node) => {
      result.push(node.value);
      if (node.left) traverse(node.left);
      if (node.right) traverse(node.right)
    }
    traverse(this.root)
    return result
  }

  DFSPostorder() {
    let result = [];

    const traverse = (node) => {
      if (node.left) traverse(node.left);
      if (node.right) traverse(node.right)
      result.push(node.value);
    }
    traverse(this.root)
    return result
  }

  DFSInoder() {
    let result = [];

    const traverse = (node) => {
      if (node.left) traverse(node.left);
      result.push(node.value);
      if (node.right) traverse(node.right)
    }
    traverse(this.root)
    return result
  }

  findSecondMostNode() {
    if (!this.root) return undefined;

    let secondMax;
    let curr = this.root;
    let prev = null;

    while (curr.right !== null) {
      prev = curr;
      curr = curr.right;
    }

    if (curr.left !== null) {
      curr = curr.left;

      while (curr.right !== null) {
        curr = curr.right;
        secondMax = curr.value
      }
    } else {
      if (curr === this.root && prev === null) {
        secondMax = undefined
      } else {
        secondMax = prev.value
      }
    }
    return secondMax
  }


  isBalanced() {
    const getHeight = (node) => {
      if (!node) return 0;
      return Math.max(getHeight(node.left), getHeight(node.right)) + 1;
    }

    const helper = (root) => {
      if (!root) return true;
      // For a given node we will get the height of its left and right sub tree.
      let heightDifference = Math.abs(getHeight(root.left) - getHeight(root.right));

      //If the height difference is greater than one then we will return false. We will repeat this process for each of the node in the tree using recursion.
      if (heightDifference > 1) {
        return false;
      } else {
        return helper(root.left) && helper(root.right);
      }
    }
    return helper(this.root)
  }

  findClosestValueInBst(target, closest = Infinity) {
    let currentNode = this.root;

    while (currentNode !== null) {
      if (Math.abs(target - closest) > Math.abs(target - currentNode.value)) {
        closest = currentNode.value;
      }
      if (target < currentNode.value) currentNode = currentNode.left;
      else if (target > currentNode.value) currentNode = currentNode.right;
      else break;
    }
    return closest
  }
}


/*
BINARY HEAPS:
  insertion O(log N)
  extraction: O(log N)
  search O (N)
*/

class MaxBinaryHeap {
  constructor() {
    this.values = [];
  }

  insert(val) {
    this.values.push(val);

    let currInd = this.values.length - 1;
    let parentInd = Math.floor((currInd - 1) / 2)

    while (val > this.values[parentInd]) {
      this.values[currInd] = this.values[parentInd]
      this.values[parentInd] = val;

      currInd = parentInd;
      parentInd = Math.floor((currInd - 1) / 2)
    }
    return this
  }

  extractMax() {
    let removed = this.values[0];
    let last = this.values.pop()

    if (this.values.length) {
      this.values[0] = last;
      this.sinkDown();
    }
    return removed
  }

  sinkDown() {
    let ind = 0;
    const length = this.values.length;
    let element = this.values[0];

    while (true) {
      let leftChildInd = ind * 2 + 1;
      let rightChildInd = ind * 2 + 2;
      let leftChild, rightChild;
      let swap = null;

      if (leftChildInd < length) {
        leftChild = this.values[leftChildInd]
        if (leftChild > element) {
          swap = leftChildInd
        }
      }

      if (rightChildInd < length) {
        rightChild = this.values[rightChildInd];
        if ((swap === null && rightChild > element) ||
          (swap !== null && rightChild > leftChild)) {
          swap = rightChildInd;
        }
      }

      if (swap === null) break;

      this.values[ind] = this.values[swap];
      this.values[swap] = element;
      ind = swap
    }
  }

  heapSort() {
    let k = this.values.length - 1;
    let sorted = [];

    while (k > 0) {
      sorted.push(this.extractMax())
    }
    this.values = sorted.reverse();
    return this;
  }

  heapifyMax() {
    const heapify = (arr, length, i) => {
      let largest = i;
      let left = i * 2 + 1;
      let right = i * 2 + 2;

      if (left < length && arr[left] > arr[largest]) largest = left;

      if (right < length && arr[right] > arr[largest]) largest = right;

      if (largest !== i) {
        [arr[i], arr[largest]] = [arr[largest, arr[i]]];

        heapify(arr, length, largest);
      }
      return arr;
    }

    let length = this.values.length;
    let parentInd = Math.floor(length / 2 - 1);

    while (parentInd > 0) {
      heapify(this.values, length, parentInd);
      parentInd--;
    }
    return this;
  }
}


/*
maxHeapify compares three elements (a parent and two children) and makes sure that they are in the correct order for a max heap. This function is going to take in three arguments: the array, the length of the array that we want to build our heap from, and the index of the parent that we are heapifying.
*/
const heapify = (arr, length, i) => {
  let largest = i;
  let left = i * 2 + 1;
  let right = i * 2 + 2;

  if (left < length && arr[left] > arr[largest]) largest = left;

  if (right < length && arr[right] > arr[largest]) largest = right;

  if (largest !== i) {
    [arr[i], arr[largest]] = [arr[largest, arr[i]]];

    heapify(arr, length, largest);
  }
  return arr;
}

const heapSort = (arr) => {
  let length = arr.length;
  let parentInd = Math.floor(length / 2 - 1);
  let k = length - 1;

  while (parentInd > 0) {
    heapify(arr, length, parentInd);
    parentInd--;
  }

  while (k > 0) {
    [arr[0], arr[k]] = [arr[k], arr[0]]
    heapify(arr, k, 0);
    k--;
  }
  return arr;
}


class PriorityQueueNode {
  constructor(val, priority) {
    this.value = val;
    this.priority = priority
  }
}

class PriorityQueue {
  constructor() {
    this.values = []
  }

  enqueue(val, priority) {
    let newNode = new PriorityQueueNode(val, priority);
    this.values.push(newNode);
    this.bubbleUp();
  }

  bubbleUp() {
    let idx = this.values.length - 1;
    const element = this.values[idx];
    while (idx > 0) {
      let parentIdx = Math.floor((idx - 1) / 2);
      let parent = this.values[parentIdx];
      if (element.priority >= parent.priority) break;
      this.values[parentIdx] = element;
      this.values[idx] = parent;
      idx = parentIdx;
    }
  }

  dequeue() {
    const min = this.values[0];
    const end = this.values.pop();
    if (this.values.length > 0) {
      this.values[0] = end;
      this.sinkDown();
    }
    return min;
  }

  sinkDown() {
    let idx = 0;
    const length = this.values.length;
    const element = this.values[0];
    while (true) {
      let leftChildIdx = 2 * idx + 1;
      let rightChildIdx = 2 * idx + 2;
      let leftChild, rightChild;
      let swap = null;

      if (leftChildIdx < length) {
        leftChild = this.values[leftChildIdx];
        if (leftChild.priority < element.priority) {
          swap = leftChildIdx;
        }
      }
      if (rightChildIdx < length) {
        rightChild = this.values[rightChildIdx];
        if (
          (swap === null && rightChild.priority < element.priority) ||
          (swap !== null && rightChild.priority < leftChild.priority)
        ) {
          swap = rightChildIdx;
        }
      }
      if (swap === null) break;
      this.values[idx] = this.values[swap];
      this.values[swap] = element;
      idx = swap;
    }
  }
}


/*
HASH TABLES
  insert: O(1)
  deletrtion: O(1)
  accsess: O(1)
*/

class HashTable {
  constructor(size = 10) {
    this.keyMap = new Array(size)
  }

  _hash(key) {
    let total = 0;
    let PRIME = 31;
    for (let i = 0; i < Math.min(key.length, 100); i++) {
      let char = key[i];
      let value = char.charCodeAt(0) - 96
      total = (total * PRIME + value) % this.keyMap.length;
    }
    return total;
  }

  set(key, val) {
    let index = this._hash(key);

    if (!this.keyMap[index]) this.keyMap[index] = [];

    this.keyMap[index].push([key, val])
  }

  get(key) {
    let index = this._hash(key);

    if (this.keyMap[index]) {
      for (let hashedKey of this.keyMap[index]) {
        if (hashedKey[0] === key) return hashedKey[1]
      }
      return undefined;
    }
  }

  keys() {
    let keys = [];
    for (let bucket of this.keyMap) {
      if (bucket) {
        for (let val of bucket) {
          if (!keys.includes(val[0])) {
            keys.push(val[0])
          }
        }
      }
    }
    return keys
  }

  values() {
    let values = [];
    for (let bucket of this.keyMap) {
      if (bucket) {
        for (let val of bucket) {
          if (!values.includes(val[1])) {
            values.push(val[1])
          }
        }
      }
    }
    return values
  }
}

/*
GRAPH
insertion:
extraction:
search
*/

class Graph {
  constructor() {
    this.adjacencyList = {}
  }

  addVertex(vertex) {
    if (!this.adjacencyList[vertex]) this.adjacencyList[vertex] = [];
    else throw alert(`${vertex} already exists`)
  }

  addEdge(vertex1, vertex2) {
    this.adjacencyList[vertex1].push(vertex2);
    this.adjacencyList[vertex2].push(vertex1);
  }

  removeEdge(vertex1, vertex2) {
    this.adjacencyList[vertex1] = this.adjacencyList[vertex1].filter(val => val !== vertex2);
    this.adjacencyList[vertex2] = this.adjacencyList[vertex2].filter(val => val !== vertex1);
  }

  removeVertex(vertex) {
    this.adjacencyList[vertex].forEach(val => this.removeEdge(val, vertex));
    delete this.adjacencyList[vertex];
  }

  DFSRecursive(vertex) {
    let result = [];
    let visited = {};
    const adjacencyList = this.adjacencyList;

    const DFS = (v) => {
      if (!v) return null;
      visited[v] = true;
      result.push(v)
      adjacencyList[v].forEach(val => {
        if (!visited[val]) return DFS(val)
      })
    }
    DFS(vertex);

    return result;
  }

  DFSIterative(start) {
    const stack = [start];
    const result = [];


    visited[start] = true;
    while (stack.length) {
      currentVertex = stack.pop();
      result.push(currentVertex);

      this.adjacencyList[currentVertex].forEach(neighbor => {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          stack.push(neighbor)
        }
      });
    }
    return result;
  }

  BFS(start) {
    let queue = [start];
    let visited = {};
    let currentVertex;
    let result = []
    visited[start] = true;

    while (queue.length) {
      currentVertex = queue.shift();
      result.push(currentVertex)

      this.adjacencyList[currentVertex].forEach(val => {
        if (!visited[val]) {
          visited[val] = true;
          queue.push(val);

        }
      })
    }
    return result
  }

  isCiclic(vertex, visited = {}, recStack = {}) {
    if (!visited[vertex]) {
      visited[vertex] = true;
      recStack[vertex] = true;
      this.adjacencyList[vertex].forEach(val => {
        let currentVertex = val;
        if (!visited[currentVertex] && this.isCiclic(currentVertex, visited, recStack)) return true;
        else if (recStack[currentVertex]) return true
      })
    }
    recStack[vertex] = false;
    return false;
  }
}

/*
Dijkstra Alghoritm: helps to find the shortest way to one vertex to another using the priority Queue I implemented above
*/
class WeightedGraph {
  constructor() {
    this.adjacencyList = {};
  }
  addVertex(vertex) {
    if (!this.adjacencyList[vertex]) this.adjacencyList[vertex] = [];
  }
  addEdge(vertex1, vertex2, weight) {
    this.adjacencyList[vertex1].push({ node: vertex2, weight });
    this.adjacencyList[vertex2].push({ node: vertex1, weight });
  }

  Dijkstra(start, finish) {
    const priorityQueue = new PriorityQueue();
    const distances = {};
    const previous = {};
    let path = [] //to return at end
    let smallest;
    //build up initial state
    for (let vertex in this.adjacencyList) {
      if (vertex === start) {
        distances[vertex] = 0;
        priorityQueue.enqueue(vertex, 0);
      } else {
        distances[vertex] = Infinity;
        priorityQueue.enqueue(vertex, Infinity);
      }
      previous[vertex] = null;
    }
    // as long as there is something to visit
    while (priorityQueue.values.length) {
      smallest = priorityQueue.dequeue().val;
      if (smallest === finish) {
        //WE ARE DONE
        //BUILD UP PATH TO RETURN AT END
        while (previous[smallest]) {
          path.push(smallest);
          smallest = previous[smallest];
        }
        break;
      }
      if (smallest || distances[smallest] !== Infinity) {
        for (let neighbor in this.adjacencyList[smallest]) {
          //find neighboring node
          let nextNode = this.adjacencyList[smallest][neighbor];
          //calculate new distance to neighboring node
          let candidate = distances[smallest] + nextNode.weight;
          let nextNeighbor = nextNode.node;
          if (candidate < distances[nextNeighbor]) {
            //updating new smallest distance to neighbor
            distances[nextNeighbor] = candidate;
            //updating previous - How we got to neighbor
            previous[nextNeighbor] = smallest;
            //enqueue in priority queue with new priority
            priorityQueue.enqueue(nextNeighbor, candidate);
          }
        }
      }
    }
    return path.concat(smallest).reverse();
  }
}

/*
TRIE
*/

class Trie {
  constructor() {
    this.characters = {};
    this.isWord = false;
  }
  addWord(word, index = 0) {
  }
}


// ALGO PROBLEMS:


// Given a sorted Arr and a number, wright a function called sortedFrequency that countshe occurences of the number in the array O(log N)
const binarySearch = (arr, left, right, target) => {
  if (right < left) return -1;

  let mid = left + Math.floor((right - left) / 2);

  // If the element is present at the middle itself
  if (arr[mid] === target) return mid;

  // If element is smaller than mid, then it can only be present in left subarray
  if (arr[mid] > target) return binarySearch(arr, left, mid - 1, target);

  // Else the element can only be present in right subarray
  return binarySearch(arr, mid + 1, right, target);
}

const sortedFrequency = (arr, target) => {
  // find the taget number in the array
  let ind = binarySearch(arr, 0, arr.length - 1, target);

  // If element is not present
  if (ind === -1) return -1;

  // Count elements on left side.
  let count = 1;
  let left = ind - 1;
  while (left >= 0 && arr[left] === target) {
    count++;
    left--;
  }

  // Count elements on right side.
  let right = ind + 1;
  while (right < arr.length && arr[right] === target) {
    count++;
    right++;
  }

  return count;
}


// Given an Arr of 1s and 0s which has all 1s first followed by all 0s, write a function called countZeroes, which returns the number of zeroes in Arr O(log N)
const firstZero = (arr, low, high) => {
  if (high >= low) {
    // Check if mid element is first 0
    let mid = low + Math.floor((high - low) / 2);

    if ((mid === 0 || arr[mid - 1] === 1) && arr[mid] === 0)
      return mid;

    // If mid element is not 0
    if (arr[mid] === 1) return firstZero(arr, (mid + 1), high);
    // If mid element is 0, but not first 0
    else return firstZero(arr, low, (mid - 1));
  }
  return -1;
}

const countZeroes = (arr) => {
  let first = firstZero(arr, 0, arr.length - 1)

  return first === -1 ? 0 : arr.length - first
}


// Write a function called findRotatedIndex which accepts a rotated array of sorted numbers and an integer. The function should return the index of the integerin the array. If the value is not found, return -1
function findRotatedIndex(arr, target, left = 0, right = arr.length - 1) {
  if (right < left) return -1;

  let mid = left + Math.floor((right - left) / 2);

  if (arr[mid] === target) return mid;
  if (arr[mid + 1] === target) return mid + 1;
  if (arr[mid - 1] === target) return mid - 1;

  if (arr[mid < target]) return findRotatedIndex(arr, target, left, right = mid - 1);
  else return findRotatedIndex(arr, target, left = mid + 1, right);
}


//Given an unsorted array and a number n, find if there exists a pair of elements in the array whose difference is n.
const findPair = (arr, n) => {
  // Initialize positions of two elements
  arr.sort((a, b) => a - b)
  let i = 0;
  let j = 1;

  // Search for a pair
  while (i < arr.length && j < arr.length) {
    if (i != j && arr[j] - arr[i] === n) return true;
    else if (arr[j] - arr[i] < n) j++;
    else i++;
  }
  return false;
}


// Find three largest numbers in the ARR:
function findThreeLargestNumbers(arr) {
  let result = [null, null, null];
  for (let num of arr) {
    updateLargest(result, num);
  }
  return result
}

function updateLargest(arr, num) {
  if (!arr[2] || num > arr[2]) shiftAndUpdate(arr, num, 2);
  else if (!arr[1] || num > arr[1]) shiftAndUpdate(arr, num, 1);
  else if (!arr[0] || num > arr[0]) shiftAndUpdate(arr, num, 0);
}

function shiftAndUpdate(arr, num, idx) {
  for (let i = 0; i <= idx; i++) {
    if (i === idx) arr[i] = num;
    else arr[i] = arr[i + 1]
  }
}


// Find the indexes of 2 values in the array which sum gives a targetSum:
function twoNumberSum(arr, targetSum) {
  let left = 0;
  let right = arr.length - 1;

  arr = arr.sort((a, b) => a - b)

  while (left < right) {
    let currentSum = arr[left] + arr[right]
    if (currentSum === targetSum) return [arr[left], arr[right]];
    if (currentSum < targetSum) left++;
    else right--
  }
  return []
}

// Find the indexes of 3 values in the array which sum gives a targetSum:
function threeNumberSum(array, targetSum) {
  array.sort((a, b) => a - b);
  let currentSum;
  let result = [];

  for (let i = 0; i < array.length - 2; i++) {
    let left = i + 1;
    let right = array.length - 1;
    while (left < right) {
      currentSum = array[i] + array[left] + array[right];
      if (currentSum === targetSum) {
        result.push([array[i], array[left], array[right]])
        left++;
        right--;
      } else if (currentSum < targetSum) {
        left++;
      } else if (currentSum > targetSum) {
        right--;
      }
    }

  }
  return result
}

// Write a function that returns the sum of the numbers in the array and, if the value of the array is another array multiply the sum of that array values by depth of this array:
function productSum(arr, depth = 1) {
  let sum = 0;
  for (let val of arr) {
    if (Array.isArray(val)) {
      sum += productSum(val, depth + 1)
    } else {
      sum += val;
    }
  }
  return sum * depth
}


// Calculate waterArea
function waterArea(blocks) {
  const rightMaxes = [];
  let rightMax = 0;
  for (let i = blocks.length - 1; i >= 0; i--) {
    rightMax = Math.max(rightMax, blocks[i]);
    rightMaxes[i] = rightMax;
  }

  const leftMaxes = [];
  let leftMax = 0;
  for (let i = 0; i < blocks.length; i++) {
    leftMax = Math.max(leftMax, blocks[i]);
    leftMaxes[i] = leftMax;
  }

  return blocks.reduce((waterCollected, block, idx) => {
    const leftMax = leftMaxes[idx];
    const rightMax = rightMaxes[idx];
    return waterCollected + Math.min(leftMax, rightMax) - block;
  }, 0);
}


// Removed the Kth node from the end in a single LL:
class LinkedList {
  constructor(value) {
    this.value = value;
    this.next = null;
  }
}

function removeKthNodeFromEnd(head, k) {
  let current = head;
  let previous = head;
  let removed;

  for (let i = 0; i < k; i++) {
    current = current.next
  }

  if (!current) {
    removed = previous;
    head.value = head.next.value;
    head.next = head.next.next;
    return removed
  }

  while (current.next) {
    previous = previous.next;
    current = current.next;
  }

  removed = previous.next;
  previous.next = previous.next.next;
  return removed
}


// Given an array of integers. Return the array of it's all possible premutations:
function getPermutations(array) {
  let result = [];
  helper(array, [], result)
  return result
}

function helper(arr, perm, result) {
  if (!arr.length && perm.length) {
    result.push(perm);
  } else {
    for (let i = 0; i < arr.length; i++) {
      let newArr = [...arr.slice(0, i), ...arr.slice(i + 1)];
      let newPerm = [...perm, arr[i]];
      helper(newArr, newPerm, result)
    }
  }
}

// Given two unsorted arrays find a pair of intergers which difference the closest to zero:
function smallestDifference(arr1, arr2) {
  arr1.sort((a, b) => a - b);
  arr2.sort((a, b) => a - b)
  let smallestPair = [];
  let i = 0;
  let j = 0;
  let smallest = Infinity;
  let current = Infinity;
  while (i < arr1.length && j < arr2.length) {
    let first = arr1[i];
    let second = arr2[j]
    if (first < second) {
      current = second - first;
      i++;
    } else if (first > second) {
      current = first - second;
      j++;
    } else {
      return [arr1[i], arr2[j]]
    }

    if (smallest > current) {
      smallestPair = [first, second];
      smallest = current;
    }
  }
  return smallestPair
}


//You are given a binary tree DS. Whire a function that checks if it's a valid BST:
function validateBst(tree) {
  const validateBSThelper = (node, min, max) => {
    if (!node) return true;
    if (node.value < min || node.value >= max) return false;
    let left = helper(node.left, min, node.value);
    let right = helper(node.right, node.value, max);
    return left && right
  }
  return validateBSThelper(tree, -Infinity, Infinity)
}


//You're given an array of integers. Whrite a function that moves all the instances of that integer in the array  to the end of the array:
function moveElementToEnd(arr, toMove) {
  let left = 0;
  let right = arr.length - 1;

  while (left < right) {
    if (arr[left] === toMove && arr[right] !== toMove) {
      swap(arr, left, right);
      right--;
      left++;
    } else if (left < right && arr[right] === toMove) {
      right--;
    } else {
      left++;
    }
  }
  return arr
}

function swap(arr, i, j) {
  [arr[i], arr[j]] = [arr[j], arr[i]];
}

//Wright a fanction that inverts a given binary tree:
function invertBinaryTree(tree) {
  const queue = [tree];
  while (queue.length) {
    let current = queue.shift();
    if (!current) continue
    swapBST(current);
    queue.push(current.left);
    queue.push(current.right);
  }
}

function swapBST(tree) {
  const left = tree.left;
  tree.left = tree.right;
  tree.right = left;
}
