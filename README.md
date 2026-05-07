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

# Notes:
#AS stieltjes
#Topk stieljes 
#e^{1/(lambda - x_i)}
#\sum_q 1/q! X 1/(lambda - x_i)^q
#CLRS bench
#TODO:
#triton
Next steps:
1) Try Adaptive Temperature Stieltjes for all the fixed q
2) Try ASStieltjes…
3) top k followed by stieltjes (for all the fixed q)
4) exponential stieltjes
—
5) If done with these on the max retrieval task, then can try the CLRS Algorithm task or any of the other tasks in the ASEntmax paper