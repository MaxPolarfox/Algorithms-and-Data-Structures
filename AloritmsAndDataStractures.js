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
    let currentVal = arr[i]

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
}
