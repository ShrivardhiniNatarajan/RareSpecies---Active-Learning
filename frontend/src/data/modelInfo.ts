export interface Round3ResultRow {
  strategy: string;
  accuracy: number;
  macroF1: number;
  mountainLionRecall: number;
}

export const DATASET_INFO = {
  name: "Caltech Camera Traps (Six-Species Subset)",
  classes: ["opossum", "coyote", "deer", "raccoon", "bobcat", "mountain_lion"],
  sourceImages: 4895,
  animalCrops: 6558,
  imagesWithAnimals: 4671,
  splits: {
    train: 4617,
    val: 971,
    test: 970,
    strategy: "Sequence-aware (no seq_id overlap across splits)"
  }
};

export const CLASSIFIER_INFO = {
  architecture: "ResNet-50 (ImageNet-pretrained)",
  optimizer: "AdamW",
  learningRate: "1e-4",
  batchSize: 32,
  epochs: "Up to 10 epochs (best-validation-accuracy checkpoint)",
  inputResolution: "224 × 224"
};

export const ACTIVE_LEARNING_INFO = {
  seedSize: 944,
  poolSize: 3673,
  budgetPerRound: 150,
  selectionUnit: "150 sequences per round (at most one crop per sequence)",
  rounds: 3,
  finalLabeledBudget: 1394,
  labeledPercentage: "30.2% of training set"
};

export const FULL_DATA_REFERENCE = {
  accuracy: 82.06,
  macroF1: 81.78,
  mountainLionRecall: 74.29
};

export const ROUND3_RESULTS_TABLE: Round3ResultRow[] = [
  {
    strategy: "Random",
    accuracy: 78.35,
    macroF1: 74.68,
    mountainLionRecall: 57.14
  },
  {
    strategy: "Uncertainty",
    accuracy: 77.63,
    macroF1: 74.90,
    mountainLionRecall: 60.00
  },
  {
    strategy: "Rarity-weighted",
    accuracy: 77.63,
    macroF1: 75.53,
    mountainLionRecall: 57.14
  }
];

export const HONEST_NOTE = 
  "No strategy was best on every metric; results come from a single run; mountain lion has 35 test crops, so one crop = 2.86 points; DAAC is an exploratory score and has not been validated as an error detector.";
