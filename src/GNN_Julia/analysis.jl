using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using DataFrames
include("../extracting_data/extract_software_metadata.jl")
include("../extracting_data/extract_article_metadata.jl")
#using .HeteroDataProcessing
using .DataReaderswMATH
import .DataReaderswMATH: generate_software_dataframe
import .DataReaderszbMATH: extract_articles_metadata
using BSON
using GraphNeuralNetworks
using SparseArrays
using Distances
using Base.Threads
using Serialization

using NPZ
filtered_msc = deserialize("filtered_msc.jls")


#articles_list_dict = extract_articles_metadata()

## 146346 lines
input2pap= hcat(permutedims(filtered_msc), grouped_by_paper_id.paper_id, software_arrays)
vi = Vector{typeof(input2pap[1, 64])}()
vs = Vector{typeof(input2pap[1, 65])}()

# Create dictionaries for faster lookups
input2_dict_64 = Dict((input2pap[line, 1:63]...,) => input2pap[line, 64] for line in 1:size(input2pap, 1))
input2_dict_65 = Dict((input2pap[line, 1:63]...,) => input2pap[line, 65] for line in 1:size(input2pap, 1))


t=DataFrame(permutedims(Matrix(g.target[:,g.ndata.test_mask])),:auto)
f=DataFrame(permutedims(Matrix(g.features[:,g.ndata.test_mask])),:auto)
# Iterate through `f` and use the dictionaries to find matches
for line in 1:size(f, 1)
    key = (f[line, 1:63]...,)  # Create a tuple key for the dictionary
    if haskey(input2_dict_64, key)
        push!(vi, input2_dict_64[key])
        push!(vs, input2_dict_65[key])
    end
end

# 66777 lines
# Track unique identifiers using a built-in set
new_values = Vector{Union{Int, Nothing}}(undef, length(test_mask))
new_eval_mask = zeros(Int, length(test_mask)) 
unique_identifiers = Set{Int}()
vi_index = 1  
for i in 1:length(test_mask)
    if test_mask[i] == 0
        new_values[i] = nothing
    elseif test_mask[i] == 1
        identifier = vi[vi_index]
        vi_index += 1
        if !in(identifier, unique_identifiers)
            new_values[i] = identifier
            new_eval_mask[i] = 1
            push!(unique_identifiers, identifier)
        else
            new_values[i] = identifier
            new_eval_mask[i] = 0
        end
    end
end

# Step 2: Verify the sum of new_eval_mask is equal to the number of unique values in vi
@assert sum(new_eval_mask) == length(unique(vi))
#true_input_target=hcat(Matrix(f),vi,vs)




t=DataFrame(permutedims(Matrix(g.target[:,BitVector(new_eval_mask)])),:auto)

f=DataFrame(permutedims(Matrix(g.features[:,BitVector(new_eval_mask)])),:auto)

#df_keywords = unique(software_df[:,[:id,:keywords]])

BSON.@load "model.bson" model opt


function final_resut(threshold)
    pred=model(g,g.features)
    pred = pred .> threshold
    pred=pred[:,BitVector(new_eval_mask)]

    pred=permutedims(pred)

    precision = sum((pred .& BitMatrix(Matrix(t)))) / (sum(pred) + 1e-6)

    # Recall: count true positives over actual positives
    recall = sum((pred .& BitMatrix(Matrix(t))) / (sum(BitMatrix(Matrix(t))) + 1e-6))

    # F1-score: harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    # True positives (TP) and true negatives (TN)
    true_positives = sum(pred .& BitMatrix(Matrix(t)))
    true_negatives = sum(.~pred .& .~BitMatrix(Matrix(t)))

    # Total number of elements
    total_elements = length(pred)

    # Accuracy: (TP + TN) / total
    accuracy = (true_positives + true_negatives) / (total_elements + 1e-6)

    return pred, precision, recall, f1_score, accuracy
end
pred, precision, recall, f1_score, accuracy = final_resut(0.1)
println("Metrics:\n - Accuracy: $accuracy\n - Precision: $precision\n - Recall: $recall\n - F1 Score: $f1_score")





###### Exporting the data to compute other models in Pythong
npzwrite("X_train.npy", Matrix(g.features[:, g.train_mask])')
npzwrite("y_train.npy", Matrix(g.target[:, g.train_mask])')
npzwrite("X_test.npy", Matrix(g.features[:,BitVector(new_eval_mask)])')
npzwrite("y_test.npy", Matrix(g.target[:,BitVector(new_eval_mask)])')


### Another anaylysis with threshold 0.4
pred, precision, recall, f1_score, accuracy = final_resut(0.4)

## Need to compute from simplegnn.jl and data_processing.jl id_to_classification,value_to_index, enc, label_to_msc
###############
 # Reverse the dictionary
 # Reverse the dictionary with sorted keys
function reverse_dict(dict)
    reversed_dict = Dict{String, Vector{Int64}}()  # Use a Vector to store multiple keys for the same value

    for (key, value) in dict
        # Split the value by ";", sort it, and join it back into a sorted string
        sorted_value = join(sort(split(value, ";")), ";")

        # Insert into the reversed dictionary
        if haskey(reversed_dict, sorted_value)
            push!(reversed_dict[sorted_value], key)  # Append the key to the existing list
        else
            reversed_dict[sorted_value] = [key]  # Create a new entry for this value
        end
    end

    return reversed_dict
end

msc2soft = reverse_dict(id_to_classification)


function compare_input_output(pap)
    function decode_msc_input(sparse_vector, value_to_index, enc)
        # Step 1: Find active indices (non-zero entries in the sparse vector)
        active_indices = findall(x -> x != 0.0, sparse_vector)

        # Step 2: Map active indices to original integer MSC codes
        index_to_value = Dict(v => k for (k, v) in value_to_index)
        decoded_integers = map(x -> index_to_value[x], active_indices)

        # Step 3: Map integers back to original MSC strings using `enc.invlabel`
        reverse_invlabel = Dict(v => k for (k, v) in enc.invlabel)
        decoded_strings = map(x -> reverse_invlabel[x], decoded_integers)

        return decoded_strings
    end

    sparse_vector = g.ndata[:features][:, pap]  # Sparse vector for the first node

    # Assuming `value_to_index` and `enc` are already defined
    decoded_msc_codes_input = decode_msc_input(sparse_vector, value_to_index, enc)



    # Decoding function to map a single row or full matrix back to MSC labels
    function decode_msc_output(msc_soft_hot, label_to_msc)
        # Invert the label_to_msc dictionary to map indices back to MSC labels
        msc_to_label = Dict(v => k for (k, v) in label_to_msc)

        # If msc_soft_hot is a vector (single row), handle it separately
   
pid= readlines("msc_paper_id.txt")

msc_pid_dict = Dict(joined_sorted_unique_prefixes_per_line[i] => pid[i] for i in 1:length(pid))


          decoded_msc = [msc_to_label[idx] for idx in true_indices]
        else
            # Handle the full matrix case (more than 1 dimension)
            decoded_msc = Vector{Vector{String}}(undef, size(msc_soft_hot, 1))
            for paper_idx in 1:size(msc_soft_hot, 1)
                true_indices = findall(msc_soft_hot[paper_idx, :])
                decoded_msc[paper_idx] = [msc_to_label[idx] for idx in true_indices]
            end
        end

        return decoded_msc
    end
    # Call the decode_msc function

   
    
    #println("Paper inputs: $decoded_msc_codes_input MSC codes")
    decoded_labels_pred_ouput = sort(decode_msc_output(pred[pap,:], label_to_msc))
    decoded_labels_true_ouput= sort(decode_msc_output(BitVector(g.ndata[:target][:,BitVector(new_eval_mask)][:,pap]), label_to_msc))
    #println("Software MSC codes associated with the paper: $decoded_labels_true_ouput MSC codes")
    #println("Software MSC codes predicted for the paper: $decoded_labels_pred_ouput MSC codes")
    #println(all(x -> x in list2, list1)
    key = join(decoded_labels_true_ouput, ";")
    println(key)
    println(msc2soft[key])
    
    return decoded_msc_codes_input, decoded_labels_true_ouput, decoded_labels_pred_ouput , msc2soft[key]
end



println(compare_input_output(811))

#### We asked ChatGPT generating the output in fugure 5 with the prompt :
#Can your make a beautiful graph linking with:
#(SubString{String}["62"], ["03", "05", "15", "49", "60", "62", "65", "68", "90", "91", "92", "93", "94"], ["00", "01", "03", "05", "11", "13", "14", "15", "26", "30"  …  "82", "83", "85", "86", "90", "91", "92", "93", "94", "97"], [11354])
#the first list is research paper MSC code, second list the true MSC of the software, the third list the MSC we predict, the last list the software id
