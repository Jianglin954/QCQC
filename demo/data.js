/*
 * Data extracted verbatim from Table 1 of
 * "Seeing Through Words: Controlling Visual Retrieval Quality with Language Models"
 * (iclr2026/iclr2026_conference.tex). Corpus: MS-COCO.
 *
 * Each entry: prefix query -> {low, medium, high} quality profile, where a profile holds
 *   completion : the continuation the LLM generates under that quality control
 *   image      : the top-1 image the completed query retrieves
 *   aes        : aesthetic score  (LAION aesthetic predictor, 1-10)
 *   rel        : relevance score  (CLIP image-text cosine similarity, 0-1)
 */
const CORPUS = "MS-COCO";

const DB = [
  {
    prefix: "a train",
    profiles: {
      low: {
        completion: "that is sitting on the tracks in gravel",
        image: "images/000000330316.jpg",
        aes: 4.715, rel: 0.347
      },
      medium: {
        completion: "sitting on the tracks with black smoke coming out of it",
        image: "images/000000161161.jpg",
        aes: 4.818, rel: 0.382
      },
      high: {
        completion: "is traveling near some water and houses",
        image: "images/000000565166.jpg",
        aes: 5.935, rel: 0.394
      }
    }
  },
  {
    prefix: "a teddy bear",
    profiles: {
      low: {
        completion: "wearing eye glasses and laying on a bed",
        image: "images/000000171309.jpg",
        aes: 4.788, rel: 0.359
      },
      medium: {
        completion: "that is sitting on a tree",
        image: "images/000000173647.jpg",
        aes: 5.649, rel: 0.388
      },
      high: {
        completion: "sitting on a wall next to an old stone house",
        image: "images/000000395505.jpg",
        aes: 5.818, rel: 0.437
      }
    }
  },
  {
    prefix: "a chair",
    profiles: {
      low: {
        completion: "with wires on it",
        image: "images/ac55472c0cd1c55ca0f33b0173d802d5.jpg",
        aes: 4.019, rel: 0.362
      },
      medium: {
        completion: "with red and black ropes on it",
        image: "images/ddf8c66a50995a092482c646868a042f.jpg",
        aes: 4.847, rel: 0.379
      },
      high: {
        completion: "on a stage in a field",
        image: "images/c9b43096e00e7ff78ecf6671832e75cc.jpg",
        aes: 5.257, rel: 0.387
      }
    }
  },
  {
    prefix: "a bowl",
    profiles: {
      low: {
        completion: "of soup with meat and vegetables in it",
        image: "images/8a927707ae96d2e83768a652ea8006d1.jpg",
        aes: 4.648, rel: 0.343
      },
      medium: {
        completion: "on display",
        image: "images/aa17fad3be64b544bc4c7598e745e3bd.jpg",
        aes: 4.980, rel: 0.379
      },
      high: {
        completion: "with flowers on it",
        image: "images/171d48be7810498bbae7928ec9f00068.jpg",
        aes: 5.386, rel: 0.387
      }
    }
  }
];

const LEVELS = ["low", "medium", "high"];
const LEVEL_LABEL = { low: "Low", medium: "Medium", high: "High" };

if (typeof module !== "undefined") { module.exports = { DB, LEVELS, LEVEL_LABEL, CORPUS }; }
