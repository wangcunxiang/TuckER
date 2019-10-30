from load_data import DataText, read_json
import time
from collections import defaultdict
from models.model import *
from torch.optim.lr_scheduler import ExponentialLR
from models.Mean import MeanTuckER
import argparse
import torch.tensor


class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0., maxlength=25, vocab_size=40452):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.maxlength = maxlength
        self.textdata = None  # = Etextdata + Rtextdata; np.array()
        self.Etextdata = None
        self.Rtextdata = None
        self.Evocab = ['NULL', ]  # padding_idx=0
        self.Rvocab = ['NULL', ]  # padding_idx=0
        self.vocab_size = vocab_size
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}

    # def strings_to_ids(self, data, vocab=['NULL', ]):#padding_idx=0; designed for [triples, sentences, words]
    #     data_ids = []
    #     for triple in data:
    #         triple_ids = []
    #         for i in triple:
    #             words = i.strip().split()
    #             word_ids = []
    #             for word in words:
    #                 if word not in vocab:
    #                     vocab.append(word)
    #                 word_ids.append(vocab.index(word))
    #             triple_ids.append(word_ids)
    #         data_ids.append(triple_ids)
    #     return data_ids, vocab

    def strings_to_ids(self, data, vocab=['NULL', ]):  # padding_idx=0; designed for [sentences, words]
        tmp = []
        for sent in data:
            sent = sent.strip().split()
            tmp += sent
        vocab += sorted(list(set(tmp)))

        vocab_ = {vocab[i]: i for i in range(len(vocab))}
        data_ids = []
        for sent in data:
            sent = sent.strip().split()
            word_ids = [vocab_[word] for word in sent]
            data_ids.append(word_ids)
        return data_ids, vocab

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])
        losses = []

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
        test_er_vocab = self.get_er_vocab(self.get_data_idxs(data))
        test_er_vocab_pairs = list(test_er_vocab.keys())  # list [...,(e1,r),...]

        print("Number of data points: %d" % len(test_data_idxs))


        for i in range(0, len(test_er_vocab_pairs), self.batch_size):
            data_batch, targets = self.get_batch(er_vocab, test_er_vocab_pairs, i)

            e1_idx = torch.LongTensor(self.Etextdata[data_batch[:, 0]])
            r_idx = torch.LongTensor(self.Rtextdata[data_batch[:, 1]])

            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                #e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            targets_ = targets.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(np.isin(sort_idxs[j], np.where(targets_[j] == 1.0)[0]))[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

            if self.label_smoothing:
                targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
            loss = model.loss(predictions, targets)
            losses.append(loss.item())

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
        print("loss="+str(np.mean(losses)))

    def check_textdata(self):
        for i in range(0, len(self.Etextdata)):
            # print(self.Etextdata[i])
            if self.Etextdata[i].all() != self.textdata[i].all():
                print(i)

    def train_and_eval(self):
        print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        # data_idxs = self.get_data_idxs(d.data)
        print("Number of training data points: %d" % len(train_data_idxs))
        # print("Number of all data points: %d" % len(data_idxs))

        ########
        # data_ids, self.vocab = self.strings_to_ids(vocab=self.vocab, data=d.data)
        #print('d.entities='+str(len(d.entities)))
        entities_ids, self.Evocab = self.strings_to_ids(vocab=self.Evocab, data=d.entities)

        #print("entities_ids = " + str(entities_ids))
        relation_ids, self.Rvocab = self.strings_to_ids(vocab=self.Rvocab, data=d.relations)
        print("entities_ids len=%d" % len(entities_ids))
        print("relation_ids len=%d" % len(relation_ids))
        print("read vocab ready.")
        d.Etextdata = d.get_index(entities_ids, self.maxlength)  # list, contained padding entities
        self.Etextdata = np.array(d.Etextdata)
        d.Rtextdata = d.get_index(relation_ids, self.maxlength)
        self.Rtextdata = np.array(d.Rtextdata)
        # self.textdata = np.array(d.Etextdata + d.Rtextdata)
        #self.check_textdata()
        print("text data ready")
        es_idx = torch.LongTensor(self.Etextdata)
        if self.cuda:
            es_idx = es_idx.cuda()
        model = MeanTuckER(d, es_idx, self.ent_vec_dim, self.rel_vec_dim, Evocab=len(self.Evocab),
                           Rvocab=len(self.Rvocab), n_ctx=self.maxlength, **self.kwargs)  # n_ctx = 52为COMET中计算出的
        print("model ready")

        ########
        if self.cuda:
            model.cuda()
        #model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)  # dict (e1,r)->e2
        er_vocab_pairs = list(er_vocab.keys())  # list [...,(e1,r),...]

        print("Starting training...")

        for it in range(1, self.num_iterations + 1):
            hits = []
            ranks = []
            for i in range(10):
                hits.append([])
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            # print(er_vocab_pairs[:])

            for j in range(0, len(er_vocab_pairs), self.batch_size):

                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                # target: tensor [batch, len(d.entities), 0./1.]
                opt.zero_grad()

                e1_idx = torch.LongTensor(self.Etextdata[data_batch[:, 0]])
                r_idx = torch.LongTensor(self.Rtextdata[data_batch[:, 1]])
                #e2_idx = torch.LongTensor(data_batch[:, 2])  # e2 are not used for model forward

                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    #e2_idx = e2_idx.cuda()\
                if e1_idx.size(0) == 1:
                    print(j)
                    continue
                predictions = model.forward(e1_idx, r_idx)
                #print("predictions="+str(predictions))

                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
                #print("sort_values="+str(sort_values))

                sort_idxs = sort_idxs.cpu().numpy()
                targets_ = targets.cpu().numpy()
                for k in range(data_batch.shape[0]):
                    rank = np.where(np.isin(sort_idxs[k], np.where(targets_[k] == 1.0)[0]))[0][0]
                    ranks.append(rank + 1)


                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
                        else:
                            hits[hits_level].append(0.0)

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))

                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())



            if self.decay_rate:
                scheduler.step()
            print(it)
            print(time.time() - start_train)
            print("loss="+str(np.mean(losses)))
            model.eval()
            with torch.no_grad():
                # print("Validation:")
                # self.evaluate(model, d.valid_data)
                if not it % 2:
                    print("Train:")
                    start_test = time.time()
                    self.evaluate(model, d.train_data)
                    print(time.time() - start_test)
                    print("Test:")
                    start_test = time.time()
                    self.evaluate(model, d.test_data)
                    print(time.time() - start_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                        help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--config", type=str, default="config/config.json", nargs="?",
                        help="the config file path")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                        help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                        help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                        help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                        help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                        help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                        help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                        help="Amount of label smoothing.")
    parser.add_argument("--max_length", type=int, default=15, nargs="?",
                        help="Batch size.")
    parser.add_argument("--vocab_size", type=int, default=40542, nargs="?",
                        help="Batch size.")
    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = DataText(data_dir=data_dir, reverse=False)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr,
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1,
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing,
                            maxlength=args.max_length
                            )
    experiment.train_and_eval()


