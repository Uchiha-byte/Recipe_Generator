# import the necessary libraries

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from Foodimg2Ing.args import get_parser
import pickle
from Foodimg2Ing.model import get_model
from torchvision import transforms
from Foodimg2Ing.utils.output_utils import prepare_output
from PIL import Image
import time
from Foodimg2Ing import app


def output(uploadedfile):

    # Keep all the codes and pre-trained weights in data directory
    data_dir = os.path.join(app.root_path, 'data')

    # code will run in gpu if available and if the flag is set to True, else it will run on cpu
    use_gpu = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'

    # Load vocabulary files
    try:
        with open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb') as f:
            ingrs_vocab = pickle.load(f)
        with open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb') as f:
            vocab = pickle.load(f)
    except Exception as e:
        print(f"Error loading vocabulary files: {str(e)}")
        raise

    ingr_vocab_size = len(ingrs_vocab)
    instrs_vocab_size = len(vocab)
    output_dim = instrs_vocab_size

    # Initialize model
    t = time.time()
    import sys; sys.argv=['']; del sys
    args = get_parser()
    args.maxseqlen = 15
    args.ingrs_only = False
    model = get_model(args, ingr_vocab_size, instrs_vocab_size)
   
    # Load the pre-trained model parameters
    model_path = os.path.join(data_dir, 'modelbest.ckpt')
    try:
        # Try different loading methods
        try:
            # Method 1: Direct load
            checkpoint = torch.load(model_path, map_location=map_loc)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e1:
            print(f"First loading attempt failed: {str(e1)}")
            try:
                # Method 2: Legacy loading
                checkpoint = torch.load(model_path, map_location=map_loc, pickle_module=pickle)
                model.load_state_dict(checkpoint)
            except Exception as e2:
                print(f"Second loading attempt failed: {str(e2)}")
                # Method 3: Last resort - try with encoding specification
                with open(model_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location=map_loc, encoding='utf-8')
                model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

    model.to(device)
    model.eval()
    model.ingrs_only = False
    model.recipe_only = False

    # Image transformation setup
    transf_list_batch = []
    transf_list_batch.append(transforms.ToTensor())
    transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406), 
                                                (0.229, 0.224, 0.225)))
    to_input_transf = transforms.Compose(transf_list_batch)

    # Process the image
    try:
        img = Image.open(uploadedfile)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        raise

    # Image transformations
    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)
    
    image_transf = transform(img)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)

    # Generate recipes
    greedy = [True, False]
    beam = [-1, -1]
    temperature = 1.0
    numgens = len(greedy)
    show_anyways = False  # if True, it will show the recipe even if it's not valid

    title = []
    ingredients = []
    recipe = []
    
    try:
        for i in range(numgens):
            with torch.no_grad():
                outputs = model.sample(image_tensor, greedy=greedy[i], 
                                    temperature=temperature, beam=beam[i], true_ingrs=None)
                    
            ingr_ids = outputs['ingr_ids'].cpu().numpy()
            recipe_ids = outputs['recipe_ids'].cpu().numpy()
                    
            outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)
                
            if valid['is_valid'] or show_anyways:
                title.append(outs['title'])
                ingredients.append(outs['ingrs'])
                recipe.append(outs['recipe'])
            else:
                title.append("Not a valid recipe!")
                recipe.append("Reason: " + valid['reason'])
                ingredients.append([])
    except Exception as e:
        print(f"Error generating recipe: {str(e)}")
        raise
            
    return title, ingredients, recipe
