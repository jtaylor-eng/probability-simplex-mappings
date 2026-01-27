# Repository Structure
 * main.py: Run an experiment from here.
 * max_retrieval_architecture/
   - architecture.py: NN architecture needed in the max retrieval problem. Allows drop in softmax replacements as found in mappings/
 * mappings/
   - adaptive_temperature.py: Adaptive temperature softmax. temperature set as a polyfit of logits Shannons entropy.
   - alpha_entmax.py: Simplex mapping based of Tsallis entropies.
   - base_cls.py: Simplex mappings are subclasses of this.
   - softmax.py: Traditional softmax.
   - sparsemax.py: Sparsemax as proposed by Martins.
   - stieltjes.py: Simplex mapping based off of Stieltjes transform.
   - type_enum.py: Enum for mapping classes. Needed in max_retrieval_architecture/architecture.py
 * dataset_gen/
   - gen.py: Fills data/ with a torch dataset to be used for max retrieval training. Specify max_block_size.
      - Note: ensure pathing to ../data correct

# TODO
 - argparser in main