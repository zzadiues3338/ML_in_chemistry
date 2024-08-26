# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 22:20:09 2024

@author: 14055
"""
from tdc.multi_pred import DTI
from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from DeepPurpose import dataset



data = DTI(name = 'DAVIS')
data.convert_to_log(form = 'binding')
split = data.get_split(method = 'cold_split', column_name = 'Drug')

X_drug = data.entity1
X_target = data.entity2
y = data.y

# Type in the encoding names for drug/protein.
drug_encoding, target_encoding = 'CNN', 'Transformer'

# Data processing, here we select cold protein split setup.
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='cold_protein', 
                                frac=[0.7,0.1,0.2])

# Generate new model using default parameters; also allow model tuning via input parameters.
config = generate_config(drug_encoding, target_encoding,
                         transformer_n_layer_target = 8,
                         batch_size=64,
                         train_epoch =5)

net = models.model_initialize(**config)

%matplotlib auto
# Train the new model.
# Detailed output including a tidy table storing validation loss, metrics, AUC curves figures and etc. are stored in the ./result folder.
net.train(train, val, test)

'''
# or simply load pretrained model from a model directory path or reproduced model name such as DeepDTA
net = models.model_pretrained(MODEL_PATH_DIR or MODEL_NAME)
'''

# Repurpose using the trained model or pre-trained model
# In this example, loading repurposing dataset using Broad Repurposing Hub and SARS-CoV 3CL Protease Target.
X_repurpose, drug_name, drug_cid = load_broad_repurposing_hub(SAVE_PATH)
target, target_name = load_SARS_CoV_Protease_3CL()

_ = models.repurpose(X_repurpose, target, net, drug_name, target_name)

# Virtual screening using the trained model or pre-trained model 
X_repurpose, drug_name, target, target_name = ['CCCCCCCOc1cccc(c1)C([O-])=O', ...], ['16007391', ...], ['MLARRKPVLPALTINPTIAEGPSPTSEGASEANLVDLQKKLEEL...', ...], ['P36896', 'P00374']

_ = models.virtual_screening(X_repurpose, target, net, drug_name, target_name)



def download_BindingDB(path = './data'):

	print('Beginning to download dataset...')

	if not os.path.exists(path):
	    os.makedirs(path)

	try:
	    url = "https://www.bindingdb.org/bind/downloads/" + [url.split('/')[-1] for url in re.findall(
		    r'(/rwd/bind/chemsearch/marvin/SDFdownload.jsp\?download_file=/bind/downloads/BindingDB_All_.*?\.tsv\.zip)',
			requests.get("https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp").text)][0]
	except Exception:
	    print("Failed to retrieve current URL for BindingDB, falling back on hard-coded URL")
	    url = "https://www.bindingdb.org/bind/downloads/BindingDB_All_202406_tsv.zip"
	saved_path = wget.download(url, path)

	print('Beginning to extract zip file...')
	with ZipFile(saved_path, 'r') as zip:
	    zip.extractall(path = path)
	    print('Done!')
	path = path + '/BindingDB_All_202406.tsv'
	return path











import torch
from torchviz import make_dot




dummy_drug = torch.randn(1, 63, 100)  # Example shape for the CNN input
dummy_target = torch.randint(0, 4114, (1, 500))  # Example shape for the Transformer input

# Perform a forward pass (will be used to create the graph)
output = net(dummy_drug, dummy_target)

# Generate the graph
graph = make_dot(output, params=dict(model.named_parameters()))