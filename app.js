import { pipeline, env } from "@xenova/transformers";

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

// Due to a bug in onnxruntime-web, we must disable multithreading for now.
// See https://github.com/microsoft/onnxruntime/issues/14445 for more information.
// env.backends.onnx.wasm.numThreads = 1;

 class PipelineSingleton {
   static task = "feature-extraction";
   static model = "Supabase/gte-small";
   static instance = null;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      this.instance = pipeline(this.task, this.model, { progress_callback });
    }

    return this.instance;
  }
}

let model = null;
let progress = null;

/**
 * Asynchronously loads the model.
 *
 * This function gets the pipeline instance which will load and build the model when run for the first time.
 * It also provides a way to track the progress of the pipeline creation, which can be used to update a UI element like a progress bar.
 *
 * @async
 * @function
 * @throws {Error} If the model cannot be loaded, an error is thrown with a message detailing the reason.
 *
 * @example
 * try {
 *   await loadModel();
 * } catch (error) {
 *   console.error(error);
 * }
 */

async function loadModel() {
  try {
    // Get the pipeline instance. This will load and build the model when run for the first time.
    model = await PipelineSingleton.getInstance((data) => {
      // You can track the progress of the pipeline creation here.
      // e.g., you can send `data` back to the UI to indicate a progress bar
      // can be accessed via vagueFinder.getProgress()
      progress = data;
    });
  } catch (error) {
    throw new Error(`Unable to load Model due to ${error}`);
  }
}

/**
 * Throws an error indicating that the model has not been loaded.
 *
 *
 * @function
 * @throws {Error} Always throws an error indicating that the model has not been loaded.
 */

function modelNotLoadedErrorMessage() {
  throw new Error("Model has not been loaded, use vagueFinder.loadModel()");
}

/**
 * Asynchronously classifies the similarity between two sentences.
 *
 * This function takes two sentences and their respective embeddings and cache flags as input.
 * It calculates the embeddings for the sentences if they are not cached.
 * Then, it calculates the cosine similarity between the two embeddings.
 * It returns an object containing the two sentences, their similarity score, and the embedding of the first sentence.
 *
 * @async
 * @function
 * @param {string} sentenceOne - The first sentence to be compared.
 * @param {string} sentenceTwo - The second sentence to be compared.
 * @param {Array<number>} embedding1Cache - The cached embedding for the first sentence.
 * @param {boolean} doesCache1Exist - Flag indicating whether the embedding for the first sentence is cached.
 * @param {Array<number>} embedding2Cache - The cached embedding for the second sentence.
 * @param {boolean} doesCache2Exist - Flag indicating whether the embedding for the second sentence is cached.
 * @throws {Error} If the model is not loaded, an error is thrown.
 * @returns {Promise<{sentenceOne: string, sentenceTwo: string, alike: number, embedding1Cache: Array<number>}>} A Promise that resolves to an object containing the two sentences, their similarity score, and the embedding of the first sentence.
 *
 * @example
 * try {
 *   const result = await classify(sentence1, sentence2, embedding1Cache, true, embedding2Cache, false);
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

const classify = async (
  sentenceOne,
  sentenceTwo,
  embedding1Cache,
  doesCache1Exist,
  embedding2Cache,
  doesCache2Exist,
) => {
  if (!doesCache2Exist && !model) {
    modelNotLoadedErrorMessage();
    return;
  }

  let embedding1 = null;
  let embedding2 = null;

  if (doesCache1Exist) {
    embedding1 = embedding1Cache;
  } else {
    embedding1 = await model(sentenceOne, {
      pooling: "mean",
      normalize: true,
    });
  }

  if (doesCache2Exist) {
    embedding2 = embedding2Cache;
  } else {
    embedding2 = await model(sentenceTwo, {
      pooling: "mean",
      normalize: true,
    });
  }

  if (!doesCache1Exist) {
    embedding1 = Array.from(embedding1.data);
  }
  if (!doesCache2Exist) {
    embedding2 = Array.from(embedding2.data);
  }

  const similarity = calculateCosineSimilarity(embedding1, embedding2);

  let result = similarity;

  function calculateCosineSimilarity(embedding1, embedding2) {
    // Calculate dot product and magnitudes
    let dotProduct = 0;
    let magnitude1 = 0;
    let magnitude2 = 0;
    for (let i = 0; i < embedding1.length; i++) {
      dotProduct += embedding1[i] * embedding2[i];
      magnitude1 += embedding1[i] * embedding1[i];
      magnitude2 += embedding2[i] * embedding2[i];
    }
    magnitude1 = Math.sqrt(magnitude1);
    magnitude2 = Math.sqrt(magnitude2);

    // Calculate cosine similarity
    const similarity = dotProduct / (magnitude1 * magnitude2);
    return similarity;
  }

  return {
    sentenceOne: sentenceOne,
    sentenceTwo: sentenceTwo,
    alike: result,
    embedding1Cache: embedding1,
  };
};

/**
 * Asynchronously compares a sentence to an array of sentences.
 *
 * This function takes a sentence and an array of sentences, and a cache flag as input.
 * It calculates the similarity between the input sentence and each sentence in the array.
 * It returns an object containing the input sentence and the array of sentences with their similarity scores.
 *
 * @async
 * @function
 * @param {string} sentence - The sentence to be compared.
 * @param {Array<string|{sentenceTwo: string, embedding: Array<number>}>} array - The array of sentences to be compared. Each element can be a string or an object with `sentenceTwo` and `embedding` properties.
 * @param {boolean} doesCache2Exist - Flag indicating whether the embeddings for the sentences in the array are cached.
 * @throws {Error} If the model is not loaded, an error is thrown.
 * @returns {Promise<{sentenceOne: string, array: Array<{sentenceTwo: string, alike: number}>}>} A Promise that resolves to an object containing the input sentence and the array of sentences with their similarity scores.
 *
 * @example
 * try {
 *   const result = await compareSentenceToArray(sentence, array, true);
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

const compareSentenceToArray = async (
  sentence,
  array,
  doesCache2Exist = false,
) => {
  if (!doesCache2Exist && !model) {
    modelNotLoadedErrorMessage();
    return;
  }
  let cache = null;
  array = [...array]; //Creating a copy, so that we don't alter the original;
  for (let i = 0; i < array.length; i++) {
    const { sentenceTwo, alike, embedding1Cache } = await classify(
      sentence,
      array[i].sentenceTwo ? array[i].sentenceTwo : array[i],
      cache,
      i !== 0,
      array[i].embedding ? array[i].embedding : null,
      doesCache2Exist,
    );
    if (i === 0) {
      cache = embedding1Cache;
    }
    array[i] = { sentenceTwo: sentenceTwo, alike: alike };
  }

  return {
    sentenceOne: sentence,
    array: array,
  };
};

/**
 * Asynchronously compares a sentence to an array of sentences and returns the results in order of similarity.
 *
 * This function takes a sentence and an array of sentences as input. It uses the `compareSentenceToArray` function to calculate the cosine similarity between the input sentence and each sentence in the array.
 * The function then sorts the results in descending order of similarity and returns an object containing the input sentence and the sorted array of comparison results.
 *
 * @async
 * @function
 * @param {string} sentence - The sentence to compare to the array of sentences.
 * @param {Array<string>} array - The array of sentences to compare to the input sentence.
 * @returns {Promise<{sentenceOne: string, array: Array<{sentenceTwo: string, alike: number}>}>} A Promise that resolves to an object containing the input sentence and an array of objects. Each object in the array contains:
 *   - `sentenceTwo`: A sentence from the input array.
 *   - `alike`: The cosine similarity score between the input sentence and `sentenceTwo`.
 *   The array is sorted in descending order of similarity score.
 * @throws {Error} If the model has not been loaded.
 *
 * @example
 * try {
 *   const result = await arrayInOrder("This is a sentence.", ["This is another sentence.", "Yet another sentence."]);
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

const arrayInOrder = async (sentence, array) => {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }
  array = [...array]; //Creating a copy, so that we don't alter the original;
  const { sentenceOne, array: returnedArray } = await compareSentenceToArray(
    sentence,
    array,
    false,
  );

  returnedArray.sort((a, b) => b.alike - a.alike);

  return {
    sentenceOne: sentenceOne,
    array: returnedArray,
  };
};

/**
 * Returns the progress of the model loading process.
 *
 * If the model is loading, it returns an Object that represents the progress of the model loading process.
 *
 * @function
 * @returns {Object} The progress of the model loading process.
 *
 * @example
 * try {
 *   const progress = getProgress();
 *   console.log(progress);
 * } catch (error) {
 *   console.error(error);
 * }
 */

function getProgress() {
  return progress;
}

/**
 * Compares two sentences using the loaded model.
 *
 * This function takes two sentences as input and uses the `classify` function to calculate the cosine similarity between them. If the model has not been loaded, it throws an error.
 *
 * @async
 * @function
 * @param {string} sentenceOne - The first sentence to compare.
 * @param {string} sentenceTwo - The second sentence to compare.
 * @returns {Promise<{sentenceOne: string, sentenceTwo: string, alike: number}>} A Promise that resolves to an object containing:
 *   - `sentenceOne`: The first input sentence.
 *   - `sentenceTwo`: The second input sentence.
 *   - `alike`: The cosine similarity score between `sentenceOne` and `sentenceTwo`.
 * @throws {Error} If the model has not been loaded.
 *
 * @example
 * try {
 *   const result = await compareTwoSentences("This is a sentence.", "This is another sentence.");
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

async function compareTwoSentences(sentenceOne, sentenceTwo) {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }

  const { alike } = await classify(
    sentenceOne,
    sentenceTwo,
    null,
    false,
    null,
    false,
  );

  return { sentenceOne, sentenceTwo, alike };
}

/**
 * Asynchronously generates embeddings for an array of sentences.
 *
 * This function takes an array of sentences as input.
 * It generates embeddings for each sentence in the array using the model.
 * It returns an array of objects, each containing a sentence and its corresponding embedding.
 *
 * Note: This function creates a copy of the input array to avoid altering the original.
 *
 * @async
 * @function
 * @param {Array<string>} array - The array of sentences for which embeddings are to be generated.
 * @throws {Error} If the model is not loaded, an error is thrown.
 * @returns {Promise<Array<{sentenceTwo: string, embedding: Array<number>}>>} A Promise that resolves to an array of objects, each containing a sentence and its corresponding embedding.
 *
 * @example
 * try {
 *   const result = await getCached(array);
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

async function getCached(array) {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }
  array = [...array]; //Creating a copy, so that we don't alter the original;
  let returnedArray = [];
  for (let i = 0; i < array.length; i++) {
    let embedding = await model(array[i], {
      pooling: "mean",
      normalize: true,
    });
    embedding = Array.from(embedding.data);
    returnedArray[i] = { sentenceTwo: array[i], embedding: embedding };
  }

  return returnedArray;
}

/**
 * Compares a sentence to an array of cached sentences.
 *
 * @async
 * @function
 * @param {string} sentence - The sentence to compare.
 * @param {Array<{sentenceTwo: string, embedding: Array<number>}>} cachedArray - The array of cached sentences to compare against.
 * @returns {Promise<{sentenceOne: string, array: Array<{sentenceTwo: string, alike: number}>}>} An object containing the original sentence and an array of comparison results.
 *
 * @example
 * const result = await cachedCompareSentenceToArray('Hello world', cachedSentences);
 * console.log(result);
 */

async function cachedCompareSentenceToArray(sentence, cachedArray) {
  cachedArray.map((item) => {
    if (!item.sentenceTwo) {
      throw new Error(
        "Each item in the cachedArray must have a sentenceTwo property",
      );
    }
    return {
      sentenceTwo: item.sentenceTwo,
      embedding: [...item.embedding],
    };
  });
  const { sentenceOne, array: returnedArray } = await compareSentenceToArray(
    sentence,
    cachedArray,
    true,
  );

  return {
    sentenceOne: sentenceOne,
    array: returnedArray,
  };
}

/**
 * Asynchronously sorts an array of sentences based on their similarity to a given sentence.
 *
 * This function takes a sentence and an array of sentences as input.
 * It calculates the similarity between the input sentence and each sentence in the array.
 * It then sorts the array based on the similarity scores in descending order.
 * It returns an object containing the input sentence and the sorted array of sentences with their similarity scores.
 *
 * This function differs from `arrayInOrder` in that it expects the array of sentences to already have cached embeddings.
 * This function is useful when you have a large array of sentences and you want to cache their embeddings to avoid recalculating them each time you compare a new sentence to the array.
 *
 * @async
 * @function
 * @param {string} sentence - The sentence to be compared.
 * @param {Array<string|{sentenceTwo: string, embedding: Array<number>}>} cachedArray - The array of sentences to be compared. Each element is a object with `sentenceTwo` and `embedding` properties.
 * @returns {Promise<{sentenceOne: string, array: Array<{sentenceTwo: string, alike: number}>}>} A Promise that resolves to an object containing the input sentence and the sorted array of sentences with their similarity scores.
 *
 * @example
 * try {
 *   const result = await cachedArrayInOrder(sentence, array);
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

async function cachedArrayInOrder(sentence, cachedArray) {
  cachedArray.map((item) => {
    if (!item.sentenceTwo) {
      throw new Error(
        "Each item in the cachedArray must have a sentenceTwo property",
      );
    }
    return {
      sentenceTwo: item.sentenceTwo,
      embedding: [...item.embedding],
    };
  });
  const { sentenceOne, array: returnedArray } = await compareSentenceToArray(
    sentence,
    cachedArray,
    true,
  );

  returnedArray.sort((a, b) => b.alike - a.alike);

  return {
    sentenceOne: sentenceOne,
    array: returnedArray,
  };
}

/**
 * Asynchronously gets the top results from a list of sentences based on their similarity to a given sentence.
 * This function is different from `getArrayInOrder` as it limits the results to the top 'numberOfResults' items.
 *
 * @async
 * @param {string} sentence - The sentence to compare against.
 * @param {Array<string>} array - The array of sentences to compare.
 * @param {number} numberOfResults - The number of top results to return. This parameter constrains the size of the returned array.
 * @throws {Error} Will throw an error if the model is not loaded or if numberOfResults is less than or equal to 0.
 * @returns {Promise<{sentenceOne: string, array: Array<{sentenceTwo: string, alike: number}>}>} A promise that resolves to an object containing the original sentence and an array of the top results.
 * The top results array contains objects with the properties 'sentenceTwo' and 'alike', where 'sentenceTwo' is a sentence from the input array and 'alike' is its similarity score to the original sentence.
 */

async function getTop(sentence, array, numberOfResults) {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }

  if (numberOfResults <= 0) {
    throw new Error("numberOfResults is either 0 or less than 0");
  }

  const arrayCopy = [...array];
  numberOfResults = Math.min(numberOfResults, arrayCopy.length);
  const list = new LinkedListInAlikeOrder(numberOfResults);
  let cache = null;

  for (let i = 0; i < array.length; i++) {
    const { sentenceTwo, alike, embedding1Cache } = await classify(
      sentence,
      array[i],
      cache,
      i !== 0,
      null,
      false,
    );
    if (i === 0) {
      cache = embedding1Cache;
    }
    list.addNode({ sentenceTwo: sentenceTwo, alike: alike });
  }

  const resultantArray = list.getArray();
  return {
    sentenceOne: sentence,
    array: resultantArray,
  };
}

/**
 * Class representing a doubly linked list with a maximum length.
 */
class LinkedListInAlikeOrder {
  head = null;
  tail = null;
  length = 0;
  maxLength = 0;

  /**
   * Create a LinkedListInAlikeOrder.
   * @param {number} maxLength - The maximum length of the linked list.
   */
  constructor(maxLength) {
    this.maxLength = maxLength;
  }

  /**
   * Create a new node.
   * @param {Object} obj - The object to be added to the node. The object should have two properties: 'alike' and 'sentenceTwo'.
   * @return {Object} The new node.
   * @private
   */
  _node(obj) {
    return {
      ...obj,
      next: null,
      prev: null,
    };
  }

  /**
   * Convert the linked list to an array.
   * @return {Array<Object>} The array representation of the linked list.
   */
  getArray() {
    let currentNode = this.head;
    const array = [];

    while (currentNode !== null) {
      const { alike, sentenceTwo } = currentNode;
      array.push({ alike, sentenceTwo });
      currentNode = currentNode.next;
    }

    return array;
  }

  /**
   * Get the index where the new node should be inserted.
   * @param {number} alike - The 'alike' value of the new node.
   * @return {number} The index where the new node should be inserted.
   * @private
   */
  _getIndex(alike) {
    let count = 0;
    let currentNode = this.head;
    while (count < this.maxLength) {
      if (currentNode === null || currentNode.alike <= alike) {
        return count;
      }
      currentNode = currentNode.next;
      count++;
    }
    return this.maxLength;
  }

  /**
   * Add a new node to the linked list.
   * @param {Object} obj - The object to be added as a new node.
   */
  addNode(obj) {
    if (this.head === null) {
      this.head = this.tail = this._node(obj);
      this.length++;
      this._audit();
      return;
    }
    let index = this._getIndex(obj.alike);
    this._insertAtIndex(index, obj);
  }

  /**
   * Insert a new node at a specific index.
   * @param {number} index - The index where the new node should be inserted.
   * @param {Object} obj - The object to be added as a new node.
   * @private
   */
  _insertAtIndex(index, obj) {
    const newNode = this._node(obj);
    let currentNode = this.head;
    if (index === 0) {
      newNode.next = this.head;
      this.head.prev = newNode;
      this.head = newNode;
    } else if (index === this.length) {
      this.tail.next = newNode;
      newNode.prev = this.tail;
      this.tail = newNode;
    } else {
      let count = 0;
      while (count + 1 < index) {
        currentNode = currentNode.next;
        count++;
      }
      newNode.next = currentNode.next;
      currentNode.next.prev = newNode;
      newNode.prev = currentNode;
      currentNode.next = newNode;
    }
    this.length++;
    this._audit();
  }

  /**
   * Check if the linked list is longer than the maximum length.
   * If it is, delete the last node.
   * @private
   */
  _audit() {
    if (this.length > this.maxLength) {
      this._deleteLastNode();
    }
  }

  /**
   * Delete the last node of the linked list.
   * @private
   */
  _deleteLastNode() {
    const newTail = this.tail.prev;
    this.tail.prev = null;
    newTail.next = null;
    this.tail = newTail;
    this.length--;
  }
}

/**
 * The `vagueFinder` object provides a set of methods for comparing sentences using a loaded model.
 *
 * @namespace
 * @property {function} loadModel - Loads the model. See {@link loadModel}.
 * @property {function} getProgress - Returns the progress of the model loading process. See {@link getProgress}.
 * @property {function} compareTwoSentences - Compares two sentences using the loaded model. See {@link compareTwoSentences}.
 * @property {function} compareSentenceToArray - Compares a sentence to an array of sentences using the loaded model. See {@link compareSentenceToArray}.
 * @property {function} arrayInOrder - Compares a sentence to an array of sentences using the loaded model and returns the results in order of similarity. See {@link arrayInOrder}.
 * @property {function} getCached - Returns a cached array. See {@link getCached}.
 * @property {function} cachedCompareSentenceToArray - Compare a sentence to an array of cached sentences. See {@link cachedCompareSentenceToArray}.
 * @property {function} cachedArrayInOrder - Compares a sentence to an array of cached senteces and returns the results in order of similarity. See {@link cachedArrayInOrder}.
 * @property {function} getTop - Compares a sentence to an array of sentences using the loaded model and returns the top 'numberOfResults' results. The number of results is constrained by the 'numberOfResults' parameter. See {@link getTop}.
 */

const vagueFinder = {
  loadModel,
  getProgress,
  compareTwoSentences,
  compareSentenceToArray,
  arrayInOrder,
  getCached,
  cachedCompareSentenceToArray,
  cachedArrayInOrder,
  getTop,
};

export { vagueFinder };