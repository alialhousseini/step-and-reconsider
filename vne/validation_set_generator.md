In this markdown folder, we would like to create the pickle files for vne datasets. This will be similar to the pickle files in `data/` where you have 3 directories corresponding to each problem separately (tsp,jssp and cvrp)

# Overview 
In order to perform very well, I remind you that what we are doingh is to extend this repo project towards using it on a entended version of VNE (read md files in `vne/`)

Anyway, we have already did some code (external and will be added later) that train a LEHD and BQ for VNE style problem. What is still missing is to have some validation datasets so that we can test our trained models.

# Task
Your task is to read the pickle of other problems first (tsp and jssp) and understand how they have been structured and organized for the training (the tgraining happens in `core/`) and based on that we would like to have a set of functions to generate pickle data for the vne problem. 

# The pickle information
The pickle files will contains the information about a substrate network and a SFC-chain like virtual network as described in $PROBLEM_FORMULATION.md and this requires using it and understanding it very carefully. 
In this way we can generate as much as we can of data and validate our work. Please consider that we can easily customize the parameters of Substrate network, virutal network and so on...

# Behavior
Your are an intelligent and helpful coding assitant with expertise in the domain and with ultrathink capabilities that you will use them extensively to deliver correct code and results, you can ask whenever you feel unconfident thus maximimzing your correct hits.


-----------------------------------------------------------------------------------------------------------

1. Gumbeldore Route

Purpose
Generate new VNE instances, solve them with search, save pseudo-expert labels, then train on those generated labels.
Before training starts, `vne_main.py` also ensures that `config.validation_set_path`
exists. If `data/vne/vne_validation_dataset.pickle` is missing, it is generated
with `vne/validation_set_generator.py`.

Script Order
vne_main.py
  -> ensure_required_datasets
    -> ensure validation pickle exists
  -> core/train.py::main_train_cycle
    -> vne_main.py::train_for_one_epoch_gumbeldore
      -> vne_main.py::get_gumbeldore_dataloader
        -> core/gumbeldore_dataset.py::GumbeldoreDataset.generate_dataset
          -> vne_main.py::generate_instances
            -> vne/instance_generator.py::make_dataset
              -> make_instance
                -> make_substrate_instance
                -> make_virtual_request
                -> _has_feasible_embedding

          -> core/gumbeldore_dataset.py::async_sbs_worker
            -> vne/trajectory.py::Trajectory.init_batch_from_instance_list
            -> vne/trajectory.py::Trajectory.log_probability_fn
              -> vne/network.py::VNEPolicyNetwork.forward
            -> vne/trajectory.py::Trajectory.transition_fn
            -> vne/trajectory.py::Trajectory.to_max_evaluation_fn

          -> vne_main.py::beam_leaves_to_result
          -> vne_main.py::save_search_results_to_dataset

        -> vne/dataset.py::RandomVNEDataset
          -> candidate_paths_from_instance
          -> build_vne_state_input

      -> vne_main.py::train_with_dataloader
        -> vne/network.py::VNEPolicyNetwork.forward
        -> CrossEntropyLoss
        -> optimizer.step
What Happens
instance_generator.py creates raw problem instances.
Each instance contains:

{
    "substrate": ...,
    "requests": [...]
}
trajectory.py searches for good embeddings.
It chooses paths in this order:

request 0, link 0
request 0, link 1
...
request 1, link 0
request 1, link 1
...
vne_main.py::beam_leaves_to_result saves the best search result:
{
    "processing_paths": ...,
    "f_placements": ...,
    "objective": ...
}
dataset.py replays the saved solution and creates supervised-style training items.
Each item is:

{
    "state": ...,
    "next_action_idx": ...
}
train_with_dataloader trains the network using cross entropy.


2. Supervised Route
Purpose
Use a solved dataset. If the canonical supervised pickle is missing,
`vne_main.py` generates it before the dataloader is built. No search-generated
Gumbeldore data is created during supervised training.

Script Order
vne_main.py
  -> ensure_required_datasets
    -> ensure validation pickle exists
    -> ensure supervised training pickle exists
  -> core/train.py::main_train_cycle
    -> vne_main.py::get_supervised_dataloader
      -> vne/dataset.py::RandomVNEDataset
        -> load config.training_set_path
        -> candidate_paths_from_instance
        -> build_vne_state_input

    -> vne_main.py::train_for_one_epoch_supervised
      -> vne_main.py::train_with_dataloader
        -> vne/network.py::VNEPolicyNetwork.forward
        -> CrossEntropyLoss
        -> optimizer.step
What Happens
dataset.py loads a solved pickle file from:
config.training_set_path
The pickle must already contain solved VNE instances with:
"processing_paths"
"f_placements"
"objective"
RandomVNEDataset replays each solved instance.
It creates one training item per:

(instance_idx, request_idx, processing_link_idx)
train_with_dataloader trains the network exactly like in the Gumbeldore route.

3. Main Difference
Gumbeldore
generate raw instances
-> search for solutions
-> save generated solutions
-> replay generated solutions
-> train
Supervised
load existing solved solutions
-> replay existing solutions
-> train
So the key difference is:

Gumbeldore creates labels during training.
Supervised uses labels that already exist.

4. Validation Route
Validation happens in both modes after each epoch.

Script Order
core/train.py::main_train_cycle
  -> vne_main.py::validate
    -> vne_main.py::evaluate
      -> load config.validation_set_path
      -> core/gumbeldore_dataset.py::GumbeldoreDataset.generate_dataset
        -> vne/trajectory.py::Trajectory.init_batch_from_instance_list
        -> search with current network
        -> vne_main.py::beam_leaves_to_result
      -> compute mean objective
Validation does not train the model. It only searches with the current network and reports objective.

5. Test Route
Testing happens after training finishes.

Script Order
core/train.py::main_train_cycle
  -> load best_model.pt
  -> vne_main.py::test
    -> vne_main.py::evaluate
      -> load config.test_set_path
      -> search with current network
      -> compute mean objective

      
6. Offline Dataset Generation
vne/validation_set_generator.py can still be used manually, but `vne_main.py`
now calls its functions automatically for missing canonical validation and
supervised-training pickles.

It can be used manually to create solved datasets:

vne/validation_set_generator.py
  -> generate_substrate
  -> generate_request
  -> solve_instance_ilp
  -> save_dataset
Those datasets can later be used by:

config.training_set_path
config.validation_set_path
config.test_set_path
Short Summary
Gumbeldore
instance_generator.py
-> trajectory.py search
-> save generated dataset
-> dataset.py replay
-> train network
Supervised
load solved dataset
-> dataset.py replay
-> train network
Validation/Test
load validation/test dataset
-> trajectory.py search
-> report objective
