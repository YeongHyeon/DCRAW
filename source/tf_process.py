import matplotlib
matplotlib.use('Agg')
import os, inspect, math, scipy.misc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def save_image(savepath, image): scipy.misc.imsave(savepath, image)

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def make_canvas(images, size):

    h, w = images.shape[1], images.shape[2]
    canvas = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        x = int(idx % size[1])
        y = int(idx / size[1])
        ys, ye, xs, xe = y*h, y*h+h, x*w, x*w+w
        canvas[ys:ye, xs:xe] = image

    return canvas

def save_feature(feature, step, savedir=""):

    reshaped_data = np.transpose(np.asarray(feature), (0, 3, 1, 2))

    num_seq = reshaped_data.shape[0]
    num_feature = reshaped_data.shape[1]
    for s in range(num_seq):
        canvas = make_canvas(reshaped_data[s], [int(np.sqrt(num_feature)), int(np.sqrt(num_feature))])
        save_image(os.path.join("%s" %(savedir), "%06d_feat_seq%03d.png" %(step, s)), canvas)

def save_result_noseq(data, height, width, canvas_size, step, savedir=""):

    reshaped_data = np.reshape(np.asarray(data), [-1, height, width]) # squeeze channel dimension
    canvas = make_canvas(reshaped_data, [canvas_size, canvas_size])
    save_image(os.path.join("%s" %(savedir), "%06d.png" %(step)), canvas)

def save_result(seq, height, width, canvas_size, step, savedir=""):

    seq = np.asarray(seq)
    for s in range(seq.shape[0]):
        reshaped_data = np.reshape(seq[s], [-1, height, width]) # squeeze channel dimension
        canvas = make_canvas(reshaped_data, [canvas_size, canvas_size])
        save_image(os.path.join("%s" %(savedir), "%06d_seq%03d.png" %(step, s)), canvas)

def save_recon_loss(recon_tr, recon_te):

    recon_tr = np.asarray(recon_tr)
    recon_te = np.asarray(recon_te)

    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(recon_tr, label="Training")
    plt.plot(recon_te, label="Test")
    plt.ylabel("Cross Encropy Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("recon.png")
    plt.close()

    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(np.log(recon_tr), label="Training")
    plt.plot(np.log(recon_te), label="Test")
    plt.ylabel("Cross Encropy Loss (Log scale)")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("recon_log.png")
    plt.close()

    np.save("recon_tr", recon_tr)
    np.save("recon_te", recon_te)

def save_kl_loss(kl_tr, kl_te):

    kl_tr = np.asarray(kl_tr)
    kl_te = np.asarray(kl_te)

    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(kl_tr, label="Training")
    plt.plot(kl_te, label="Test")
    plt.ylabel("Kullback-Leibler Divergence")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("kl.png")
    plt.close()

    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(np.log(kl_tr), label="Training")
    plt.plot(np.log(kl_te), label="Test")
    plt.ylabel("KL Divergence (Log scale)")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("kl_log.png")
    plt.close()

    np.save("kl_tr", kl_tr)
    np.save("kl_te", kl_te)

def training(sess, neuralnet, saver, dataset, epochs, batch_size, canvas_size, sequence_length, print_step=1):

    print("\n* Training to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    make_dir(path="recon_tr_gt")
    make_dir(path="recon_tr")
    make_dir(path="recon_te_gt")
    make_dir(path="recon_te")
    make_dir(path="enc_tr")
    make_dir(path="dec_tr")

    x_tr_static, _ = dataset.next_train(canvas_size**2)
    x_te_static, _ = dataset.next_test(canvas_size**2)

    train_writer = tf.summary.FileWriter(PACK_PATH+'/Checkpoint')
    iterations = int(dataset.num_tr/batch_size)
    not_nan = True
    list_recon_tr, list_recon_te, list_kl_tr, list_kl_te = [], [], [], []
    for epoch in range(epochs+1):
        if((epoch % print_step == 0) or (epoch == (epochs))):

            seq_tr, loss_recon_tr, loss_kl_tr = sess.run([neuralnet.recon, neuralnet.loss_recon, neuralnet.loss_kl], feed_dict={neuralnet.x:x_tr_static})
            seq_te, loss_recon_te, loss_kl_te = sess.run([neuralnet.recon, neuralnet.loss_recon, neuralnet.loss_kl], feed_dict={neuralnet.x:x_te_static})

            list_recon_tr.append(loss_recon_tr)
            list_recon_te.append(loss_recon_te)
            list_kl_tr.append(loss_kl_tr)
            list_kl_te.append(loss_kl_te)

            print("Epoch %06d (%d iteration)\nTR Loss - Recon: %.5f  KL: %.5f" % (epoch, iterations*epoch, loss_recon_tr, loss_kl_tr))
            print("TE Loss - Recon: %.5f  KL: %.5f" % (loss_recon_te, loss_kl_te))

            save_result_noseq(data=x_tr_static, height=dataset.height, width=dataset.width, canvas_size=canvas_size, step=epoch, savedir="recon_tr_gt")
            save_result(seq=seq_tr, height=dataset.height, width=dataset.width, canvas_size=canvas_size, step=epoch, savedir="recon_tr")
            save_result_noseq(data=x_te_static, height=dataset.height, width=dataset.width, canvas_size=canvas_size, step=epoch, savedir="recon_te_gt")
            save_result(seq=seq_te, height=dataset.height, width=dataset.width, canvas_size=canvas_size, step=epoch, savedir="recon_te")

            enc_feature, dec_feature = sess.run([neuralnet.enc_feat, neuralnet.dec_feat], feed_dict={neuralnet.x:x_tr_static})
            enc_feature, dec_feature = np.asarray(enc_feature), np.asarray(dec_feature)
            save_feature(feature=enc_feature[:, 0, :, :, :], step=epoch, savedir="enc_tr")
            save_feature(feature=dec_feature[:, 0, :, :, :], step=epoch, savedir="dec_tr")

        for iteration in range(iterations):
            x_tr, _ = dataset.next_train(batch_size)

            summaries = sess.run(neuralnet.summaries, feed_dict={neuralnet.x:x_tr})
            train_writer.add_summary(summaries, iteration+(epoch*iterations))

            loss_recon_tr, loss_kl_tr, _ = sess.run([neuralnet.loss_recon, neuralnet.loss_kl, neuralnet.optimizer], feed_dict={neuralnet.x:x_tr})
            # print("   [%d/%d] TR Loss - Recon: %.5f  KL: %.5f" % (iteration, iterations, loss_recon_tr, loss_kl_tr))
            if(math.isnan(loss_recon_tr) or math.isnan(loss_kl_tr)):
                not_nan = False
                break

        if(not_nan): saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")
        else:
            print("Training is terminated by Nan Loss")
            break

    save_recon_loss(list_recon_tr, list_recon_te)
    save_kl_loss(list_kl_tr, list_kl_te)

def validation(sess, neuralnet, saver, dataset, canvas_size):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    make_dir(path="recon_te_final")
    make_dir(path="enc_te_final")
    make_dir(path="dec_te_final")

    iterations = int(dataset.num_tr/canvas_size**2)
    loss_recon_tot, loss_kl_tot = [], []
    for iteration in range(iterations):
        x_te, _ = dataset.next_test(canvas_size**2)
        seq_te, loss_recon_te, loss_kl_te = sess.run([neuralnet.recon, neuralnet.loss_recon, neuralnet.loss_kl], feed_dict={neuralnet.x:x_te})
        save_result(seq=seq_te, height=dataset.height, width=dataset.width, canvas_size=canvas_size, step=iteration, savedir="recon_te_final")
        loss_recon_tot.append(loss_recon_te)
        loss_kl_tot.append(loss_kl_te)

        enc_feature, dec_feature = sess.run([neuralnet.enc_feat, neuralnet.dec_feat], feed_dict={neuralnet.x:x_te})
        enc_feature, dec_feature = np.asarray(enc_feature), np.asarray(dec_feature)
        save_feature(feature=enc_feature[:, 0, :, :, :], step=iteration, savedir="enc_te_final")
        save_feature(feature=dec_feature[:, 0, :, :, :], step=iteration, savedir="dec_te_final")

    loss_recon_tot = np.asarray(loss_recon_tot)
    loss_kl_tot = np.asarray(loss_kl_tot)
    print("Recon:%.5f  KL:%.5f" %(loss_recon_tot.mean(), loss_kl_tot.mean()))
