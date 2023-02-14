import pandas as pd
import numpy as np
import akshare as ak
import pandas_market_calendars as mcal
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
import matplotlib.lines as mlines
import seaborn as sns
import matplotlib.ticker as mticker


class SnowballEngine:
    
    def __init__(self, start_date, end_date, terms):
           
        sh = mcal.get_calendar('XSHG').schedule(start_date, end_date)
        self.df = pd.DataFrame(sh['market_open'].index, columns=['date'])
        self.df['month'] = self.df['date'].dt.month
        self.df['knockout_obs'] = self.df['month'] != self.df['month'].shift(1) 
        end_index = terms['ko_obs_protect'] + 1
        mod_idx = self.df[self.df['knockout_obs']==True].index[0:end_index]
        self.df.loc[mod_idx,'knockout_obs'] = False
        
        self.start_date = self.df.date.iloc[0]
        self.end_date = self.df.date.iloc[-1]
        self.lastdays = (self.end_date - self.start_date).days
        self.terms = terms
        
        
    def process_price(self, close):
        
        s = close / close.iloc[0]
        self.s0 = s.iloc[0]
        self.st = s.iloc[-1]
        self.df['close'] = s.values
        self.df['knockin_flag'] = self.df['close'] <= self.terms['ki_price']
        self.df['knockout_flag'] = (self.df['close'] >= self.terms['ko_price']) & (self.df['knockout_obs'])

    
    def knockout_return(self):
        
        spot = self.df.loc[self.df['knockout_flag']==True, 'close'].iloc[0]
        end_date = self.df.loc[self.df['knockout_flag']==True, 'date'].iloc[0]
        duration = (end_date - self.start_date).days / self.lastdays 
        payoff = self.terms['coupon'] * duration + self.terms['ko_participation'] * (spot/self.s0 - 1)
        
        return payoff


    def knockin_return(self):
          
        performance = self.st/(self.s0 * self.terms['ki_strike']) - 1  
        duration = self.lastdays / self.terms['annual'] 

        payoff = max(performance, -(1-self.terms['retention'])) * self.terms['ki_participation'] * duration
                            
        return payoff


    def noknockin_return(self):
        
        duration = self.lastdays / self.terms['annual']     
        # 到期敲出
        if self.st >= self.terms['ko_price']:
            payoff = self.terms['coupon'] * duration + self.terms['ko_participation'] * (self.st/self.s0 - 1)
        # 到期未敲出未敲入
        else:
            payoff = self.terms['bonus'] * duration
                
        return payoff


    def snowball_state(self):

        # 判断敲出观察日是否敲出
        if self.df['knockout_flag'].any():
            state = '到期前敲出'
            end_time = self.df.loc[self.df['knockout_flag']==True, 'date'].iloc[0]
            payoff = SnowballEngine.knockout_return(self)      
        
        # 判断是否敲入且到期  
        elif (self.df['knockin_flag'].any()) & (self.st < self.terms['ko_price']):
            state = '敲入到期'
            end_time = self.end_date
            payoff = SnowballEngine.knockin_return(self)  
                     
        # 到期未敲入
        else:
            state = '到期未敲入'
            end_time = self.end_date
            payoff = SnowballEngine.noknockin_return(self)
                
        return {'state': state, 'end_time': end_time.strftime('%Y-%m-%d'), 'payoff': payoff}
        
        
        
class KmeansEngine:
    
    def __init__(self, num_clusters, data, code):
        
        self.num_clusters = num_clusters
        self.data = data
        self.model = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", max_iter=500, random_state=10)
        self.y_pred = self.model.fit_predict(data)
        self.code = code
        
    def plot_dba_result(self):
        fig = plt.figure()
        fig.tight_layout()
        fig.suptitle(f"{self.code} 聚类结果")
        for yi in range(self.num_clusters):
            ax = plt.subplot(3, 3, yi+1)
            for xx in self.data[self.y_pred == yi].index:
                ax.plot(self.data.loc[xx], "k-", alpha=.2)
            ax.plot(self.model.cluster_centers_[yi], "r-")
            ax.set_title(f"类别 {yi+1}", fontsize=8, pad=0.5, weight='bold')
            plt.tick_params(axis='both', which='major', labelsize=3, pad=0.5)
        plt.show()
        fig.clf()



class ClusterAnalyse:
    
    
    def __init__(self, snowball_engine, terms):
        self.terms = terms
        self.snowball_engine = snowball_engine
        self.snowball_dict = dict()
        self.snowball_knockin_prob = dict()
        # self.train_cluster_center = pd.DataFrame(index=range(terms['sample_length']))
        self.train_cluster_center = dict()
        self.train_cluster_result = dict()
        self.test_cluster_result = dict()
        self.test_cluster = np.empty((0,6), float)

  
    def get_samples(self, df, code):
        
        s2 = np.array(df[['日期','收盘']])
        sample_len = self.terms['sample_length']
        backtest_start_id = np.array(range(sample_len, s2.shape[0]- self.terms['backtest_length'], self.terms['sliding_window']))
        backtest_end_id = backtest_start_id + self.terms['backtest_length']
        
        for i in range(len(backtest_start_id)):
            sbacktest = s2[backtest_start_id[i] : backtest_end_id[i], :]
            self.snowball_engine.process_price(pd.Series(sbacktest[:,1]))
            snowball_result = self.snowball_engine.snowball_state() 
            self.snowball_dict[f'{code}_{i}'] = snowball_result

        # 对雪球无法回测到的予以剔除
        s2 = s2[ : backtest_start_id[i], :]
        
        if self.terms['sliding']:
            sliding_window = self.terms['sliding_window']
            s_tmp = np.lib.stride_tricks.sliding_window_view(s2, sample_len, axis=0)[::sliding_window, :]
            s2 = s_tmp.transpose(0, 2, 1).reshape(-1, 2)
        
        # set symbol column
        scode = np.repeat(code, s2.shape[0]).reshape(-1,1)
        s2 = np.append(s2, scode, axis=1)
        # set trainid column
        tmp_s = np.zeros((s2.shape[0],2))
        # 每sample_length个交易日生成一条样本
        strainid = np.repeat(np.array(range(s2.shape[0] // sample_len)), sample_len)
        id_add = np.repeat(s2.shape[0]//sample_len, s2.shape[0]-len(strainid))
        strainid = np.append(strainid, id_add)
        tmp_s[:, 0] = strainid
        # set train order
        sorder = np.array(list(range(sample_len)) * (s2.shape[0] // sample_len))
        order_add = np.array(list(range(s2.shape[0] % sample_len)))
        trainorder = np.append(sorder,order_add)
        tmp_s[:, 1] = trainorder

        s2 = np.append(s2, tmp_s, axis=1)
            
        # set scaled price
        sprice = np.zeros((s2.shape[0], 1))
        for i in list(range(0, s2.shape[0], sample_len)):
            scaler = StandardScaler()
            scaleds = scaler.fit_transform(s2[i: i+sample_len, 1].reshape(-1,1))
            sprice[i : (i + sample_len)] = scaleds
        s2 = np.append(s2, sprice, axis=1)

        return s2
    
    
    def cluster(self, s_cluster, code, plot):
        # 以cluster的order为index，构建dataframe
        df = pd.DataFrame(s_cluster, index=s_cluster[:,4], columns=['日期','收盘','代码','聚类id','order','scaled_close'])
        df['scaled_close'] = pd.to_numeric(df['scaled_close'])
        df['聚类id'] = df['聚类id'].astype(int)
        data = df.pivot(columns='聚类id', values='scaled_close').T                            
        num_clusters = self.terms['num_clusters']        
        kmeans = KmeansEngine(num_clusters, data, code)
        for k in list(df['聚类id'].unique()):
            self.train_cluster_result[f'{code}_{k}'] = kmeans.y_pred[k]  #以code_聚类id的形式记录每一个聚类id属于哪一类
        for j in range(num_clusters):
            center = kmeans.model.cluster_centers_[j]   
            self.train_cluster_center[f'{code}_{j}'] = center.ravel()  #以code_cluster形式记录每个聚类中心的信息
        if plot:
            kmeans.plot_dba_result()


    def train(self, df, code, plot):
        if not df.empty:
            s_cluster = ClusterAnalyse.get_samples(self, df, code)
            ClusterAnalyse.cluster(self, s_cluster, code, plot)
        else:
            print('The stock data is empty!')
            
            
    def second_cluster(self, plot):
        first_cluster_center = pd.DataFrame.from_dict(self.train_cluster_center).T
        num_clusters = self.terms['num_clusters']
        title = '样本长度为' + str(self.terms['sample_length']) + '的训练集'
        self.kmeans_train = KmeansEngine(num_clusters, first_cluster_center, title)
        second_train_cluster = dict()
        if plot:
            self.kmeans_train.plot_dba_result()
        for m in range(first_cluster_center.shape[0]):
            second_train_cluster[first_cluster_center.index[m]] = self.kmeans_train.y_pred[m]           
        for i,j in self.train_cluster_result.items():
            code = i.split('_')[0]
            self.train_cluster_result[i] = second_train_cluster[code+'_'+str(j)]

            
    def get_clustern_snowball_state(self, clustern, mode):
        if mode == 'train':
            cluster_index = np.where(np.array(list(self.train_cluster_result.values())) == clustern)[0]
            clusteri = np.array(list(self.train_cluster_result.keys()))[list(cluster_index)]
        elif mode == 'test':
            cluster_index = np.where(np.array(list(self.test_cluster_result.values())) == clustern)[0]
            clusteri = np.array(list(self.test_cluster_result.keys()))[list(cluster_index)]
        snowballresult = pd.DataFrame(self.snowball_dict).T.loc[list(clusteri)]
        
        return snowballresult
    
    
    def get_knockin_prob(self, mode):
        for _ in range(self.terms['num_clusters']):
            self.snowball_result = ClusterAnalyse.get_clustern_snowball_state(self, _, mode)
            state = self.snowball_result['state'].value_counts()
            if '敲入到期' in state.index:
                percentage = state['敲入到期'] / state.sum()
            else: percentage = 0
            print(f'cluster {_+1} 敲入概率: {percentage}')
            self.snowball_knockin_prob[f'cluster {_+1}'] = percentage
            
    
    def test(self, df, code, train_engine):
        if not df.empty:
            s_cluster = ClusterAnalyse.get_samples(self, df, code)
            self.test_cluster = np.append(self.test_cluster, s_cluster, axis=0)
            for id in set(s_cluster[:,3]):
                s = s_cluster[s_cluster[:,3]==id, 5]
                distance_list = []
                for c in range(self.terms['num_clusters']):
                    dis = dtw(s.reshape(-1,1), train_engine.kmeans_train.model.cluster_centers_[c])
                    distance_list.append(dis)
                h = np.argmin(distance_list)
                self.test_cluster_result[f'{code}_{int(id)}'] = h
        else:
            print('The stock data is empty!')
    
    
    def plot_test_cluster(self, train_engine):
        fig = plt.figure()
        fig.tight_layout()
        fig.suptitle(f"样本长度为{self.terms['sample_length']}的测试集聚类结果")
        for yi in range(self.terms['num_clusters']):
            ax = plt.subplot(3, 3, yi+1)
            ind = np.where(np.array(list(self.test_cluster_result.values()))==yi)[0]
            code_id = np.array(list(self.test_cluster_result.keys()))[list(ind)]
            for xx in code_id:
                code = xx.split('_')[0]
                id = int(xx.split('_')[1])
                data = self.test_cluster[(self.test_cluster[:,2]==code)&(self.test_cluster[:,3]==id), 5]
                ax.plot(data, "k-", alpha=.2)
            ax.plot(train_engine.kmeans_train.model.cluster_centers_[yi], "r-")
            ax.set_title(f"类别 {yi+1}", fontsize=8, pad=0.5, weight='bold')
            plt.tick_params(axis='both', which='major', labelsize=3, pad=0.5)


class ResultEngine:
    
    def __init__(self, train_engine, test_engine):
        self.train_engine = train_engine
        self.test_engine = test_engine
        self.sample_len = train_engine.terms['sample_length']
        self.colors = sns.color_palette('pastel')
        
        
    def cal_sample_length(self):
        
        cluster_mapping = {1: '类别 1', 2: '类别 2', 3: '类别 3', 4: '类别 4', 5: '类别 5', 6: '类别 6', 
                       7: '类别 7', 8: '类别 8', 9: '类别 9'}
        train_cluster_array = np.array(list(self.train_engine.train_cluster_result.values())) + 1
        cluster_train = np.unique(train_cluster_array, return_counts=True)[0]
        perc_train = np.unique(train_cluster_array, return_counts=True)[1] / np.unique(train_cluster_array, return_counts=True)[1].sum()
        train_sample = pd.concat([pd.Series(cluster_train).rename('index'), pd.Series(perc_train).rename('训练集比重')], axis=1)

        test_cluster_array = np.array(list(self.test_engine.test_cluster_result.values())) + 1
        cluster_test = np.unique(test_cluster_array, return_counts=True)[0]
        perc_test = np.unique(test_cluster_array, return_counts=True)[1] / np.unique(test_cluster_array, return_counts=True)[1].sum()
        test_sample = pd.concat([pd.Series(cluster_test).rename('index'), pd.Series(perc_test).rename('测试集比重')], axis=1)
        
        df_sample = train_sample.merge(test_sample, on='index')
        df_sample = df_sample.replace({'index': cluster_mapping})
        self.df_sample = df_sample
        
        return df_sample
    
    
    def plot_cluster_num(self):
        plt.subplots()
        green = sns.color_palette('Greens')
        blue = sns.color_palette('Blues')
        ax1 = sns.barplot(data=self.df_sample, x=self.df_sample['类别'], y=self.df_sample['比重'], 
                          hue=self.df_sample['type'], palette={'训练集':green[0],'测试集':blue[1]})
        for p in ax1.patches:
            height = p.get_height()
            ax1.text(x = p.get_x()+(p.get_width()/2), 
            y = height + 0.005, 
            s = str(int(np.round(height*100,2)))+'%',
            ha = 'center', fontsize=5) 
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax1.tick_params(axis='both', which='major', labelsize=6)
        ax1.set_title(f'样本长度为{self.sample_len}的类别分布', fontsize=12)
        plt.legend(loc=1, prop={'size': 8})
        plt.show()
        
        
    def get_rank_df(self):
        
        cluster_mapping = {'cluster 1': '类别 1', 'cluster 2': '类别 2', 'cluster 3': '类别 3', 'cluster 4': '类别 4',
                   'cluster 5': '类别 5', 'cluster 6': '类别 6', 'cluster 7': '类别 7', 'cluster 8': '类别 8', 'cluster 9': '类别 9'}
        result_train = pd.DataFrame.from_dict(self.train_engine.snowball_knockin_prob, orient='index', columns=['train']).reset_index()
        result_test = pd.DataFrame.from_dict(self.test_engine.snowball_knockin_prob, orient='index', columns=['test']).reset_index()
        result = result_train.merge(result_test, on='index')
        result['train_rank'] = result['train'].rank()
        result['test_rank'] = result['test'].rank()
        result = result.replace({'index': cluster_mapping}).merge(self.df_sample, on='index')
        self.rank_result = result
        
        return result
    
    def get_mean_dict(self):
        
        test_snowball = pd.DataFrame.from_dict(self.test_engine.snowball_dict, orient='index')
        train_snowball = pd.DataFrame.from_dict(self.train_engine.snowball_dict, orient='index')
        test_mean = sum(test_snowball['state']=='敲入到期')/test_snowball.shape[0]
        train_mean = sum(train_snowball['state']=='敲入到期')/train_snowball.shape[0]
        train_mean_rank = self.rank_result.loc[np.where(self.rank_result['train'] < train_mean), 'train_rank'].max() + 0.5
        test_mean_rank = self.rank_result.loc[np.where(self.rank_result['test'] < test_mean), 'test_rank'].max() + 0.5

        mean_dict = {'test_mean': round(test_mean * 100, 2), 'train_mean': round(train_mean * 100, 2), 
               'test_rank': test_mean_rank, 'train_rank': train_mean_rank}
        self.mean_dict = mean_dict
        
        return mean_dict
        
    @staticmethod
    def newline(p1, p2, c):
        ax = plt.gca()
        l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=c, marker='o', markersize=8)
        ax.add_line(l)
        return l
    

    def plot_result_rank(self):
        df = self.rank_result
        _, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=80)
        # Vertical Lines
        ax.vlines(x=1, ymin=0, ymax=10, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
        ax.vlines(x=3, ymin=0, ymax=10, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
        # Horizontal Lines
        train_mean = self.mean_dict['train_mean']
        ax.plot(1, self.mean_dict['train_rank'], color='red', marker='o', markersize=10)
        ax.text(x=-0.1, y=self.mean_dict['train_rank']-0.1, s=f'训练集平均敲入概率：{train_mean} %', weight='bold')
        test_mean = self.mean_dict['test_mean']
        ax.plot(3, self.mean_dict['test_rank'], color='blue', marker='o', markersize=10)
        ax.text(x=3.1, y=self.mean_dict['test_rank']-0.1, s=f'测试集平均敲入概率：{test_mean} %', weight='bold')
        # Line Segmentsand Annotation
        i = 0
        for p1, p2, q1, q2, c, r1, r2 in zip(df['train_rank'], df['test_rank'], df['train'], df['test'], 
                                             df['index'], df['训练集比重'], df['测试集比重']):
                ResultEngine.newline([1, p1], [3, p2], self.colors[i])
                train_ratio = round(r1 * 100, 2)
                test_ratio = round(r2 * 100, 2)
                ax.text(1 - 0.05, p1, c + f' ({train_ratio}%), ' + str(round(q1 * 100, 2)) + '%', horizontalalignment='right', 
                        verticalalignment='center', fontdict={'size': 12})
                ax.text(3 + 0.05, p2, c + f' ({test_ratio}%), ' + str(round(q2 * 100, 2)) + '%', horizontalalignment='left', 
                        verticalalignment='center', fontdict={'size': 12})
                i += 1
        # 'Train' and 'Test' Annotations
        ax.text(1 - 0.05, df.shape[0]+0.5, '训练集', horizontalalignment='right', verticalalignment='center',
                fontdict={'size': 15, 'weight': 700})
        ax.text(3 + 0.05, df.shape[0]+0.5, '测试集', horizontalalignment='left', verticalalignment='center',
                fontdict={'size': 15, 'weight': 700})
        # Decoration
        ax.set_title(f"样本长度为{self.sample_len}的雪球敲入概率排名对比", fontdict={'size': 20})
        ax.set(xlim=(0, 4), ylim=(0, df.shape[0]+1), ylabel='敲入概率排名')
        plt.ylabel('敲入概率排名', fontsize=15)
        ax.set_xticks([1, 3])
        ax.set_xticklabels(["训练集", "测试集"], fontdict={'size': 15, 'weight': 700})
        plt.yticks(np.arange(1, df.shape[0]+1, 1), fontsize=12)
        # Lighten borders
        plt.grid(axis = 'y', alpha = 0.2)
        plt.gca().spines["top"].set_alpha(.0)
        plt.gca().spines["bottom"].set_alpha(.0)
        plt.gca().spines["right"].set_alpha(.0)
        plt.gca().spines["left"].set_alpha(.0)
        plt.show()