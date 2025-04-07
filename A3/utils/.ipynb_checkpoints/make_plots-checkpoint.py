import matplotlib.pyplot as plt

def make_plot(train_array, val_array, tokenizer_name, model_name, num_layers, use_positional, plot_dir):
    if model_name=='transformer':
        save_model_name = f'{model_name} {int(use_positional)} layers: {num_layers}' 
    else:
        save_model_name=model_name
    epochs = range(1, len(train_array) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_array, 'b-o', label='Train Metric')
    plt.plot(epochs, val_array, 'r-o', label='Validation Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')  
    plt.title(f'Training and Validation Metrics for model:{save_model_name} using tokenizer: {tokenizer_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/accuracy_plot_{save_model_name}_{tokenizer_name}.png')