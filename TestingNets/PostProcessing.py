def best_blob_fun():
    out_3dnp = val_outPost[0].detach().cpu().numpy()
    out_3dnp = out_3dnp.squeeze()
    out_3dnp = dilation(out_3dnp, ball(2))
    label_out_3dnp = label(out_3dnp)
    props = regionprops(label_out_3dnp, val_inputs[0].detach().cpu().numpy().squeeze())

    print("num de blobs predicted: ", len(props))
    for n in range(len(props)):
        r = props[n]
        # print('prediction bbox',r.bbox,"size",len(r.coords))
        patch = np.zeros(out_3dnp.shape)
        for j in range(len(r.coords)):
            patch[r.coords[j][0], r.coords[j][1], r.coords[j][2]] = 1
        # Create different matrixes, one for each blob to send to metrics
        predicted_blobn = np.zeros(out_3dnp.shape)
        predicted_blobn[label_out_3dnp == n + 1] = 1
        print("Bounding Box: ", r.bbox)
        print("Values: ", r.axis_major_length / r.axis_minor_length, "Feret: ", r.feret_diameter_max)
        print("Intensity:", r.intensity_max, r.intensity_min, r.intensity_mean)
        print("Entropy:", shannon_entropy(predicted_blobn))

        if len(props) == 1:
            bestBlob = np.expand_dims(predicted_blobn, 0)
            tensor_blobn = torch.from_numpy(bestBlob)
        elif n == 0:
            ratio_zero = r.axis_major_length / r.axis_minor_length
            minint_zero = r.intensity_min
            entr_zero = shannon_entropy(predicted_blobn)
            blob_zero = predicted_blobn
        elif n > 0:
            ratio_curr = r.axis_major_length / r.axis_minor_length
            minint_curr = r.intensity_min
            entr_curr = shannon_entropy(predicted_blobn)
            if (ratio_curr > 3) or (ratio_zero > 3):
                print("Selected by ratio")
                if ratio_curr < ratio_zero:
                    bestBlob = np.expand_dims(predicted_blobn, 0)
                    tensor_blobn = torch.from_numpy(bestBlob)
                else:
                    bestBlob = np.expand_dims(blob_zero, 0)
                    tensor_blobn = torch.from_numpy(bestBlob)
            else:
                if minint_zero < 0.0001:
                    bestBlob = np.expand_dims(predicted_blobn, 0)
                    tensor_blobn = torch.from_numpy(bestBlob)
                elif minint_curr < 0.0001:
                    bestBlob = np.expand_dims(blob_zero, 0)
                    tensor_blobn = torch.from_numpy(bestBlob)
                elif entr_curr < entr_zero:
                    bestBlob = np.expand_dims(predicted_blobn, 0)
                    tensor_blobn = torch.from_numpy(bestBlob)
                else:
                    bestBlob = np.expand_dims(blob_zero, 0)
                    tensor_blobn = torch.from_numpy(bestBlob)

