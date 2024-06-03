import { pipeline, env } from "@xenova/transformers";

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

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