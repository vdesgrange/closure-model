def training_ode_solver_net(net, epochs, t_n, x_n, dataset, val_epoch=10, rands=[], downsize=0):
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    
    tr_dataset, val_dataset = process_dataset(dataset)
    
    training_batch_size = len(tr_dataset)
    training_set_idx = np.arange(0, training_batch_size)
    
    val_batch_size = len(val_dataset)
    val_set_idx = np.arange(0, val_batch_size)
    
    tr_min_t = 1
    tr_max_t = int(len(dataset[0][0]) / 5 * 5)
    val_max_t = int(tr_max_t + len(dataset[0][0]) / 5 * 0)
    
    for e in range(1, epochs + 1):
        loss_tot = 0
        val_loss_tot = 0
        np.random.shuffle(training_set_idx)
        
        # === Train ===
        net.train()
        for i in training_set_idx:
            # === Randomness ====
            rand = -1
            if (i < len(rands)):
                rand = rands[i]
            
            optimizer.zero_grad()
            
            tr_t, tr_bu, _, _ = tr_dataset[i]
            # tr_t =  t[tr_min_t:tr_max_t]
            # tr_b0 = tr_bu[tr_min_t, :]
            # tr_bu = bu[tr_min_t:tr_max_t, :]
            tr_b0 = tr_bu[0, :]
            
            pred_u = odeint(net, tr_b0, tr_t) # [1:-1]
            loss = loss_fn(pred_u.T, tr_bu.T) # [:, 1:-1]
            loss_tot += loss.item()
            
            loss.backward(retain_graph=False) # retain_graph=True if 2+ losses
            optimizer.step()
        
        print('Epoch %d loss %f'%(e, float(loss_tot / float(training_batch_size))))
        
        # === Evaluate ===
        if (e > val_epoch):
            net.eval()
            with torch.no_grad():
                np.random.shuffle(val_set_idx)
                
                for j in val_set_idx:
                    val_t, val_bu, _, _ = dataset[j]
                    # val_t = t[tr_max_t:val_max_t]
                    # val_b0 = bu[tr_max_t, :]
                    # val_bu = bu[tr_max_t:val_max_t, :]
                    val_b0 = val_bu[0, :]

                    val_pred_u = odeint(net, val_b0, val_t)
                    val_loss = loss_fn(val_pred_u.T, val_bu.T)
                    val_loss_tot += val_loss.item()
                print('Epoch %d validation loss %f'%(e, float(val_loss_tot / float(training_batch_size))))

        if e % 10 == 0:
            sample_t, sample_b0, sample_real = get_burgers_batch(t_max, t_min, x_max, x_min, t_n, x_n, nu, rand, -1)
            sample_b0 = sample_real[0, :]
            sample_pred = odeint(net, sample_b0, sample_t)
            # sample_pred = odeint(net, sample_b0, torch.from_numpy(np.linspace(0,5,640)).float())
            show_state(sample_real[1:].T, 'Real', 't', 'x', None)
            show_state(sample_pred.detach().numpy().T, 'Determined', 't', 'x', None)
    
    return net
