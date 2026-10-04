"""Создание методологического отчета и PDF с итогами из таблиц results/."""
from pathlib import Path
import json
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor,white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfgen.canvas import Canvas
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
REPORTS=ROOT/'reports'
PALETTE=['#245A81','#239C88','#C08E3F','#AB5681','#6864A3','#65863A','#C65A43']


def fonts():
    """DejaVu Sans входит в matplotlib; отдельная установка шрифтов не нужна."""
    directory=Path(matplotlib.get_data_path())/'fonts/ttf'
    pdfmetrics.registerFont(TTFont('DV',str(directory/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DV-Bold',str(directory/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')


def footer(canvas,doc):
    canvas.setFont('DV',8); canvas.setFillColor(HexColor('#60707B'))
    canvas.drawString(42,24,'Потребительские профили муниципалитетов · СберИндекс')
    canvas.drawRightString(553,24,str(doc.page))


def load():
    tables={name:pd.read_csv(RESULTS/(name+'.csv')) for name in [
        'method_comparison','monthly_metrics','cluster_profiles','representative_municipalities',
        'assignments','transitions','stability','temporal_ablation','external_validation',
        'dictionary_sensitivity','same_month_changes','change_cases','final_profiles','sensitivity']}
    tables['run']=json.loads((RESULTS/'run.json').read_text())
    tables['quality']=json.loads((RESULTS/'data_quality.json').read_text(encoding='utf-8'))
    tables['types']=yaml.safe_load((ROOT/'configs/types.yaml').read_text(encoding='utf-8'))
    return tables


def plots(t):
    from matplotlib.colors import ListedColormap
    from matplotlib.collections import LineCollection
    from sklearn.decomposition import PCA
    from scipy import sparse
    from municipal.data import prepare,COMPOSITION
    cfg=yaml.safe_load((RESULTS/'config_used.yaml').read_text(encoding='utf-8'))
    panel,xs,info,_,_=prepare(cfg)
    months=t['run']['months']; k=t['run']['k']
    labels=t['assignments'].pivot(index='territory_id',columns='month',values='cluster').reindex(info.index)
    pca=PCA(2).fit(xs[months[0]])
    coords=np.stack([pca.transform(xs[m]) for m in months])
    final=labels[months[-1]].to_numpy()
    dest=REPORTS/'figures'; dest.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#425563','text.color':'#163244','xtick.color':'#425563','ytick.color':'#425563','figure.facecolor':'white','savefig.facecolor':'white'})
    def save(name):
        plt.savefig(dest/(name+'.png'),dpi=180,bbox_inches='tight'); plt.close()
    fig,ax=plt.subplots(figsize=(9,5))
    xy=coords[-1]; w=sparse.load_npz(RESULTS/'graphs'/f'{months[-1]}.npz').tocoo()
    take=np.flatnonzero(w.row<w.col)
    rng=np.random.default_rng(42); take=rng.choice(take,min(2200,len(take)),replace=False)
    segments=np.stack([xy[w.row[take]],xy[w.col[take]]],axis=1)
    ax.add_collection(LineCollection(segments,colors='#c2ccd0',linewidths=.35,alpha=.25))
    for c in range(1,k+1):
        ax.scatter(*xy[final==c].T,s=9,alpha=.8,color=PALETTE[c-1],label=f'Т{c}')
    ax.set_xlabel(f'Компонента 1: {pca.explained_variance_ratio_[0]:.1%} дисперсии января 2023')
    ax.set_ylabel(f'Компонента 2: {pca.explained_variance_ratio_[1]:.1%}')
    ax.legend(ncol=k,loc='upper right',frameon=False)
    ax.set_title('Декабрь 2024 · проекция признаков, не географическая карта',loc='left',fontsize=12)
    save('network')
    prof=t['cluster_profiles']; values=prof[['share_'+c for c in COMPOSITION]].to_numpy()*100
    fig,ax=plt.subplots(figsize=(9,4))
    im=ax.imshow(values,aspect='auto',cmap='YlGnBu',vmin=0,vmax=50)
    ax.set_xticks(range(6),['Продовольствие','Здоровье','Маркетплейсы','Общепит','Транспорт','Прочие'],rotation=20,ha='right')
    ax.set_yticks(range(k),[f'Т{i+1} · {int(n)} МО' for i,n in enumerate(prof['count'])])
    for i in range(k):
        for j in range(6): ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',color='white' if values[i,j]>25 else '#163244',fontsize=12)
    ax.set_title('Медианы долей расходов, % · декабрь 2024',loc='left',fontsize=12)
    fig.colorbar(im,ax=ax,fraction=.03,pad=.025)
    save('profiles')
    fig,ax=plt.subplots(figsize=(9,4.3))
    order=np.lexsort((labels.iloc[:,0],final)); arr=labels.to_numpy()[order]
    ax.imshow(arr,aspect='auto',interpolation='nearest',cmap=ListedColormap(PALETTE[:k]),vmin=1,vmax=k)
    ax.set_xticks(np.arange(0,24,3),[months[i] for i in range(0,24,3)],rotation=30,ha='right')
    ax.set_ylabel('Муниципалитеты, сгруппированные по типу в декабре 2024')
    ax.set_title('Траектории 2 016 муниципалитетов · цвет соответствует типу',loc='left',fontsize=12)
    save('trajectories')
    fig,ax=plt.subplots(figsize=(9,3.6))
    main=t['monthly_metrics'].query('method=="graph_temporal"')
    ab=t['temporal_ablation']
    ax.plot(np.arange(1,24),main.switch_rate.iloc[1:]*100,color=PALETTE[0],lw=2,label='С временным штрафом')
    ax.plot(np.arange(1,24),ab.switch_rate.iloc[1:]*100,color=PALETTE[2],lw=2,label='Без временного штрафа')
    ax.set_xticks(np.arange(1,24,3),[months[i] for i in range(1,24,3)],rotation=25,ha='right')
    ax.set_ylabel('Сменили тип, % МО'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.2)
    save('switches')
    tr=t['transitions'].groupby(['source','target'])['count'].sum().unstack().to_numpy()
    tr=tr/tr.sum(axis=1,keepdims=True)*100
    fig,ax=plt.subplots(figsize=(5.7,4.5)); im=ax.imshow(tr,cmap='Blues',vmin=0,vmax=100)
    ax.set_xticks(range(k),[f'Т{i+1}' for i in range(k)]);ax.set_yticks(range(k),[f'Т{i+1}' for i in range(k)])
    ax.set_xlabel('Тип в следующем месяце');ax.set_ylabel('Тип в текущем месяце')
    for i in range(k):
        for j in range(k):ax.text(j,i,f'{tr[i,j]:.1f}',ha='center',va='center',color='white' if tr[i,j]>50 else '#163244')
    ax.set_title('Все месячные переходы, % по строке',fontsize=12)
    save('transitions')
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    for ax,col,title,scale in zip(axes,['salary','population','market_access'],['Зарплата, тыс. руб.','Население, тыс. чел.','Доступность рынков'],[1000,1000,1]):
        ax.bar([f'Т{i+1}' for i in range(k)],prof[col]/scale,color=PALETTE[:k]); ax.set_title(title,fontsize=12);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Внешние показатели · медианы внутри типов',fontsize=13,y=1.03)
    save('external')
    fig,ax=plt.subplots(figsize=(9,3.5))
    names={'graph_temporal':'Сеть + время','kmeans':'K-means','spectral':'Спектральная','ward':'Ward'}
    groups=list(names); st=t['stability']
    for i,g in enumerate(groups):
        vals=st[st.method==g].ARI
        ax.scatter(vals,np.full(len(vals),i),s=35,color=PALETTE[i],alpha=.65)
        ax.scatter([vals.mean()],[i],marker='|',s=600,color='#172f3e',linewidths=2)
    ax.set_yticks(range(4),[names[g] for g in groups]);ax.set_xlim(0,1);ax.set_xlabel('ARI на пересечении полной выборки и подвыборки');ax.grid(axis='x',alpha=.2)
    save('stability')
    # Все значения для офлайн-просмотра встроены в HTML.
    viewer=dict(months=months,types=t['types'],colors=PALETTE[:k],variance=pca.explained_variance_ratio_.round(4).tolist(),
        coords=coords.round(3).tolist(),labels=labels.to_numpy().tolist(),
        nodes=[dict(id=int(i),name=str(row.municipal_district_name_short),region=str(row.region_name)) for i,row in info.iterrows()],
        spending=np.stack([panel.loc[m]['Все категории'].to_numpy() for m in months]).astype(int).tolist(),
        market=np.stack([panel.loc[m]['share_Маркетплейсы'].to_numpy() for m in months]).round(4).tolist(),
        food=np.stack([panel.loc[m]['share_Продовольствие'].to_numpy() for m in months]).round(4).tolist())
    template=(ROOT/'templates/explorer.html').read_text(encoding='utf-8')
    payload=json.dumps(viewer,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    (ROOT/'explorer.html').write_text(template.replace('__DATA__',payload),encoding='utf-8')


def number(x,digits=0):
    return f'{x:,.{digits}f}'.replace(',',' ').replace('.',',')


def results_text(t):
    r=t['method_comparison'].set_index('method'); p=t['cluster_profiles']; main=r.loc['graph_temporal']; km=r.loc['kmeans']
    st=t['stability'].groupby('method').ARI.mean(); reps=t['representative_municipalities']
    text=['# Результаты исследования','',
        'Пять потребительских типов выделены на полной панели 2 016 МО за 24 месяца. Число групп выбрано по калибровочным данным первого полугодия 2023 года. Все дальнейшие показатели рассчитаны на этой зафиксированной конфигурации.',
        '', '## Сравнение методов', '',
        'Средние значения за июль 2023 — декабрь 2024. Для всех методов K = 5. SW рассчитан на фиксированной случайной подвыборке до 1 200 МО в каждом месяце.', '',
        '| Метод | SW ↑ | CH ↑ | S_Dbw ↓ | AVI ↑ | AVU ↓ | MQ ↑ | Q ↑ | ARI соседних месяцев ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method,row in r.iterrows():
        text.append('| '+method+' | '+' | '.join(f'{row[c]:.3f}' for c in ['SW','CH','S_Dbw','AVI','AVU','MQ','Q','ARI_previous'])+' |')
    text+=['',f'У K-means немного выше силуэт ({km.SW:.4f} против {main.SW:.4f}) и CH. Основная модель дает более устойчивую временную последовательность: средний ARI соседних месяцев {main.ARI_previous:.3f} против {km.ARI_previous:.3f}. Универсального превосходства по всем метрикам нет. Преимущество по устойчивости во времени частично задано самой целевой функцией.',
        '',f'В десяти повторениях с подвыборкой 80% МО средний ARI равен {st.graph_temporal:.3f} для основной модели и {st.kmeans:.3f} для K-means. Здесь временной штраф выключен. По устойчивости к составу выборки K-means лучше.',
        '', '## Экономические портреты', '', 'Портреты относятся к декабрю 2024. Доли — медианы по муниципалитетам, поэтому сумма медиан отдельных категорий может отличаться от 100%. Названия описывают обнаруженные различия и не являются официальной классификацией территорий.','']
    for _,row in p.iterrows():
        c=int(row.cluster); examples=reps[reps.cluster==c].head(2)
        phrase='; '.join(f'{a["name"]} ({a["region"].strip()})' for _,a in examples.iterrows())
        text += [f'### Тип {c}. {t["types"][c]["name"]}', '',
            f'{int(row["count"])} МО. {t["types"][c]["interpretation"]} Доли продовольствия, маркетплейсов и общепита: {row["share_Продовольствие"]*100:.1f}%, {row["share_Маркетплейсы"]*100:.1f}% и {row["share_Общественное питание"]*100:.1f}%. Медианная зарплата — {number(row.salary)} руб.; индекс доступности рынков — {number(row.market_access,1)}. Близкие к центру группы примеры: {phrase}.', '']
    text += ['## Динамика и чувствительность','']
    ab=t['temporal_ablation'].query('month>"2023-06"').switch_rate.mean()
    text += [f'Средняя месячная доля смен типа на оценочном периоде составляет {main.switch_rate*100:.2f}%. При отключении временного штрафа она возрастает до {ab*100:.2f}%. Более плавная траектория удобна для сравнения территорий, но может задерживать отражение быстрого изменения профиля. Матрица переходов не описывает вероятность будущего шока.', '']
    dictionary=t['dictionary_sensitivity'].groupby('sample').mean(numeric_only=True)
    text += [f'Среди {int(dictionary.loc["single_dictionary_version","n"])} МО с единственной версией справочника средняя доля смен типа за весь период — {dictionary.loc["single_dictionary_version","switch_rate"]*100:.2f}%; для всех МО — {dictionary.loc["all","switch_rate"]*100:.2f}%. Это ограниченная проверка административных изменений: она не заменяет сопоставление исторических границ.', '']
    delta=t['same_month_changes'].query('month_of_year==12').set_index('category')
    text += [f'При сравнении декабря 2024 с декабрем 2023 медиана индивидуальных изменений доли маркетплейсов составляет +{delta.loc["Маркетплейсы","median_change_pp"]:.2f} п.п., продовольствия — {delta.loc["Продовольствие","median_change_pp"]:.2f} п.п. Это перераспределение наблюдаемой безналичной корзины. Оно не доказывает замещение конкретных офлайн-магазинов.', '']
    cases=t['change_cases'].query('dictionary_versions==1').head(3)
    for _,a in cases.iterrows():
        text += [f'- {a.municipal_district_name_short} ({a.region_name.strip()}, id {int(a.territory_id)}): тип {int(a.cluster_2023_12)} → {int(a.cluster_2024_12)}, изменение доли маркетплейсов {a.marketplace_change_pp:+.2f} п.п. между декабрями. Это один из крупнейших сдвигов в пространстве атрибутов; причина требует отдельного исследования.']
    sens=t['sensitivity']; text+=['',f'На последнем срезе при изменении правила ребра и числа соседей ARI относительно основной временной модели лежит от {sens.ARI_main.min():.3f} до {sens.ARI_main.max():.3f}. В этих экспериментах временной штраф отключен, поэтому отклонение одновременно включает эффект отсутствия временного якоря. Взаимный kNN с 10 соседями дает {int(sens[(sens.neighbors==10)&(sens.mutual==True)].iloc[0].components)} компонент, тогда как основной граф связен.', '', '## Внешняя проверка','']
    for _,a in t['external_validation'].iterrows():
        text += [f'- {a.indicator}: n = {int(a.n)}, eta² для log(1+x) = {a.eta_squared:.3f}, перестановочное p = {a.permutation_p:.3f}.']
    text += ['', 'Внешние показатели не использовались при обучении. Различия поддерживают содержательную интерпретацию типов, но не доказывают причинную связь. Перестановки не учитывают пространственную зависимость. Население не наблюдается для части МО, включая некоторые внутригородские территории, поэтому медианы рассчитаны по доступным значениям.', '', '## Использование результатов','',
        'Для выбора аналогов территории следует начать с ее типа, затем проверить близость конкретных атрибутов и контекст. Для наблюдения изменений полезно сочетать траекторию типа с непрерывными долями расходов: Катангский район, например, сильно меняет профиль между декабрями, оставаясь в типе 4. Граница кластера не является самостоятельным экономическим событием.', '',
        'Ограничения: два года наблюдений, сбалансированный отбор МО, различия цен и демографии, административные изменения и изменяющийся охват безналичных платежей. Классификация относится к потребительской стороне локальной экономики. По этим данным нельзя уверенно назвать территорию промышленной, аграрной или туристической.']
    (REPORTS/'results.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def pdf_reports(t):
    fonts(); REPORTS.mkdir(exist_ok=True)
    body=ParagraphStyle('body',fontName='DV',fontSize=9.5,leading=14,spaceAfter=8,textColor=HexColor('#243d4c'))
    title=ParagraphStyle('title',parent=body,fontName='DV-Bold',fontSize=24,leading=29,spaceAfter=18)
    heading=ParagraphStyle('heading',parent=body,fontName='DV-Bold',fontSize=15,leading=19,spaceAfter=12)
    small=ParagraphStyle('small',parent=body,fontSize=8,leading=11)
    def P(s,style=body):return Paragraph(s,style)
    def chart(name,width=500):
        from PIL import Image as PILImage
        path=REPORTS/'figures'/(name+'.png'); im=PILImage.open(path)
        return Image(str(path),width=width,height=width*im.height/im.width)
    def table(rows,widths):
        data=[[P(str(c),small) for c in row] for row in rows]
        o=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        o.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e9f0f3')),('LINEBELOW',(0,0),(-1,0),.6,HexColor('#9fb3bf')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7)])); return o
    q=t['quality']; r=t['method_comparison'].set_index('method'); main=r.loc['graph_temporal']; profiles=t['cluster_profiles']; k=t['run']['k']
    story=[P('Потребительские профили<br/>муниципалитетов',title),P('Методологический отчет и результаты · 2023–2024',heading),
        P(f'<b>{q["municipalities"]} муниципалитетов · {q["months"]} месяца · {k} типов</b>'),
        P('Цель исследования - найти сопоставимые локальные потребительские экономики и проследить изменения их состава. Сеть строится ежемесячно по сходству корзины и относительного уровня расходов.'),
        chart('network'),P('Цвет узла соответствует типу. Показана выборка 2 200 ребер для читаемости. Расчеты используют полную сеть. Оси - главные компоненты атрибутов, зафиксированные по январю 2023.',small),
        P('Ключевой результат',heading),P(f'Модель с сетевым и временным штрафами сохраняет почти тот же силуэт, что K-means ({main.SW:.3f} против {r.loc["kmeans","SW"]:.3f}), при более устойчивых месячных разбиениях (ARI {main.ARI_previous:.3f} против {r.loc["kmeans","ARI_previous"]:.3f}). По устойчивости к изменению состава выборки K-means лучше.'),PageBreak(),
        P('1. Данные и признаки',title),
        P('Источник - СберИндекс, набор Data → Sense. 303 126 строк охватывают 2 190 идентификаторов за январь 2023 - декабрь 2024. Используется полная панель 2 016 МО; 174 неполных ряда исключены без заполнения пропусков.'),
        P('Исходные файлы получены из публичной копии материалов хакатона 2025 года. В data/sources.json закреплены адрес первоисточника, commit зеркала и SHA-256. Официальный сервер при сборке недоступен, совпадение с текущей официальной версией не проверено.'),
        P('Ключ объединения - territory_id. Справочник дает названия, регион и административные версии. Используются названия на конец 2024 года. У 183 МО есть несколько версий за анализируемый интервал. Геометрическая гармонизация границ не выполнялась.'),
        P('Конструкция атрибутов',heading),
        P('Корзина состоит из продовольствия, здоровья, маркетплейсов, общепита, транспорта и остатка «Прочие расходы». Остаток равен общему показателю минус пять категорий. Во всех используемых наблюдениях он неотрицателен.'),
        P('К компонентам добавляется 0,5 единицы, затем применяется CLR: log(p_j) - среднее log(p). Отдельный признак - логарифм общего показателя за вычетом медианы по МО текущего месяца. Это относительный уровень, а не корректировка на региональные цены.'),
        P('Медианы и межквартильные размахи масштабирования оцениваются только по первому полугодию 2023 года. Значения ограничиваются [-4; 4]. CLR-блок делится на sqrt(6), относительный уровень умножается на 0,5. Все параметры заданы в YAML.'),
        P('Внешние показатели',heading),
        P('Зарплата Росстата по всем отраслям, население и доступность рынков нужны для интерпретации и не участвуют в обучении. Полностью одинаковые дубликаты удалены: 5 строк зарплаты и 8 строк населения. В населении суммируются только итоговые строки мужчин и женщин, без повторного учета возрастных групп.'),
        P('Панель сформирована ретроспективно. Полнота ряда проверяется на всех 24 месяцах, поэтому результаты не следует трактовать как проверку системы реального времени.'),PageBreak(),
        P('2. Сеть и временная модель',title),
        P('Узел - МО, ребро - сходство атрибутов. Каждый месяц выбираются 20 ближайших соседей по евклидовому расстоянию. Ребро существует при включении хотя бы в один список соседей. Матрица симметрична, петель нет.'),
        P('<b>w_ij = exp(-d_ij² / (sigma_i × sigma_j))</b><br/>sigma_i - расстояние до 20-го соседа. Локальный масштаб учитывает разную плотность наблюдений. Географическое соседство не требуется.'),
        P('Основной алгоритм минимизирует сумму трех членов:',heading),
        P('<b>J = компактность атрибутов + lambda × разрез сети + tau × смены меток</b>'),
        P('Компактность - сумма квадратов расстояний до центров, деленная на число признаков. Разрез - сумма A_ij для разных меток, где A = D^(-1/2) W D^(-1/2). Последний член - число отличий от меток предыдущего месяца. lambda = 0,5, tau = 0,15.'),
        P('Инициализация K-means с 20 перезапусками. Затем узлы последовательно переносятся только при уменьшении целевой функции; центры обновляются после прохода. Пустые группы запрещены. Не более 30 проходов. Глобальный минимум не гарантируется.'),
        P('Метки согласуются венгерским алгоритмом по максимальному пересечению групп. В первом месяце номера упорядочены по уровню расходов. Фиксированное K сохраняет набор типов; рождение и исчезновение групп отдельно не моделируются.'),
        P('Выбор параметров и сравнение',heading),
        P('K = 3…7 проверяются на январе, марте и июне 2023 года. Основная модель выбирает K по среднему рангу SW, CH, S_Dbw и Q, при среднем минимальном размере группы не менее 20. Выбрано K = 5. Коэффициенты штрафов зафиксированы заранее.'),
        P('K-means, Ward, спектральная и основная модель сравниваются при общем K на июле 2023 - декабре 2024. Все используют одинаковые атрибуты и сеть. Это сравнение общей детализации, не оптимальных настроек каждого метода.'),PageBreak(),
        P('3. Качество и устойчивость',title)]
    cols=['SW','CH','S_Dbw','AVI','AVU','MQ','Q']
    rows=[['Метод','SW ↑','CH ↑','S_Dbw ↓','AVI ↑','AVU ↓','MQ ↑','Q ↑']]
    for name,row in r.iterrows(): rows.append([name]+[f'{row[c]:.3f}' for c in cols])
    story += [table(rows,[101]+[58]*7),Spacer(1,12),
        P('SW: силуэт, подвыборка 1 200 узлов с фиксированным seed. CH: отношение межгруппового и внутригруппового разброса. S_Dbw: компактность плюс плотность между центрами. Меньшее S_Dbw предпочтительно. NaN при неопределенном знаменателе сохраняется, а не заменяется нулем.'),
        P('AVI оценивает удержание веса внутри группы, AVU - связанность разных групп относительно внешних связей. MQ рассчитан в варианте BasicMQ по плотностям связей. Q - отдельная модульность относительно конфигурационной модели. Формулы и соглашения полностью приведены в methodology.md.'),
        chart('stability'),P('Точки - 10 подвыборок по 80% МО; черта - средний ARI. Граф и модель строятся заново, временной штраф выключен. Это проверка устойчивости, не доверительный интервал для всех территорий.',small),
        P('Контроль включает 20 случайных перестановок меток с сохранением размеров групп. Вариации правила ребра и числа соседей сохранены в sensitivity.csv. При взаимном kNN с 10 соседями возникают 27 компонент, основной граф последнего месяца связен.'),PageBreak(),
        P('4. Пять типов потребительского спроса',title),chart('profiles'),
        P('Декабрь 2024. Доли рассчитаны как медианы по МО. Сумма медиан не обязана равняться 100%. Типы описывают потребление и не задают отраслевую классификацию производства.',small)]
    for _,row in profiles.iterrows():
        c=int(row.cluster); example=t['representative_municipalities'].query('cluster==@c').iloc[0]
        story.append(P(f'<b>Т{c}. {escape(t["types"][c]["name"])}</b> ({int(row["count"])} МО). {escape(t["types"][c]["interpretation"])} Пример рядом с центром группы: {escape(example["name"])} ({escape(example.region.strip())}).'))
    story += [PageBreak(),P('5. Внешний экономический контекст',title),chart('external'),
        P('Годовая зарплата, население и доступность рынков не входят в атрибуты модели. Они характеризуют найденные группы после кластеризации. Сравнение использует доступные значения; пропуски не заполняются.'),
        table([['Показатель','Наблюдений','eta², log(1+x)','p, перестановки']]+[[a.indicator,str(int(a.n)),f'{a.eta_squared:.3f}',f'{a.permutation_p:.3f}'] for _,a in t['external_validation'].iterrows()],[155,95,125,132]),Spacer(1,14),
        P('Использовано 999 перестановок меток. Пространственная зависимость в тесте не учитывается, p приведены как описательная проверка. Различия не устанавливают причинную связь.'),
        P('В типе 4 высокая зарплата сочетается с низкой доступностью рынков. Тип 5 также имеет высокую зарплату, но более крупные рынки и расширенный спрос на услуги. Это различие полезнее единственной шкалы «низкие - высокие расходы».'),
        P('Тип 1 отличается высокой долей базовых категорий и низким уровнем сервисных расходов. Типы 2 и 3 занимают промежуточное положение; у типа 3 ниже доли транспорта и здоровья. Для выбора муниципалитета-аналога нужно дополнительно сравнивать исходные показатели и местный контекст.'),PageBreak(),
        P('6. Как меняется состав групп',title),chart('trajectories'),chart('switches',width=490),
        P('Каждая горизонтальная линия верхнего рисунка - одно МО. Порядок задан типом в декабре 2024. Изменение цвета означает переход. Нижний рисунок показывает влияние временного штрафа.',small),PageBreak(),
        P('7. Интерпретация изменений',title)]
    ab=t['temporal_ablation'].query('month>"2023-06"').switch_rate.mean()
    story += [P(f'Средняя доля смен типа на оценочном периоде - {main.switch_rate*100:.2f}%, без временного штрафа - {ab*100:.2f}%. Более плавная динамика облегчает сравнение, но может задерживать фиксацию реальных изменений.'),
        P('Между декабрем 2023 и декабрем 2024 медиана индивидуальных изменений доли маркетплейсов составляет +3,55 п.п., продовольствия -3,12 п.п. Сопоставление одинаковых месяцев уменьшает влияние сезонности, но не устраняет его полностью.'),
        P('Богородский район Кировской области: тип 2 сменился типом 1, доля маркетплейсов выросла на 14,52 п.п. Ульчский район Хабаровского края: переход 2 → 1 при росте этой доли на 17,29 п.п. Катангский район Иркутской области остается в типе 4, несмотря на сильный сдвиг атрибутов. Эти примеры выбраны по величине изменения между декабрями; причины требуют отдельной проверки.'),
        P('Среди 1 833 МО с единственной версией справочника средняя доля смен типа за весь период - 9,31%, для всех МО - 9,54%. Такая проверка не заменяет гармонизацию исторических границ.'),
        P('Ограничения',heading),
        P('Всего два года данных. Классификация зависит от состава признаков и полноты рядов. Цены, демография, платежное поведение и административные изменения могут влиять на показатели. Временная гладкость задана моделью, а сетевые метрики частично отражают ее собственную цель. Переход не является доказанным структурным шоком.'),
        P('Воспроизведение',heading),P('Python 3.12. Установить requirements.txt, затем выполнить download_data.py, run.py, analyze_results.py и build_report.py. Все настройки в configs/default.yaml. Исходные файлы уже включены и проверяются по SHA-256. Результаты, сети и исходный код находятся в репозитории.'),
        P('Литература и данные',heading),
        P('СберИндекс: Data → Sense и версионный справочник МО. Zelnik-Manor, Perona (2004), Self-Tuning Spectral Clustering. Halkidi, Vazirgiannis (2001), DOI 10.1109/ICDM.2001.989517. Biswas, Biswas (2017), DOI 10.1016/j.eswa.2016.11.011. Shalileh et al. (2026), DOI 10.1134/S1064562425700589. Mancoridis et al. (1999), Bunch. Полные ссылки: methodology.md и data/sources.json.',small)]
    doc=SimpleDocTemplate(str(REPORTS/'methodology.pdf'),pagesize=(595,842),rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=42,title='Потребительские профили муниципалитетов',author='')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    presentation(t)


def presentation(t):
    c=Canvas(str(REPORTS/'presentation.pdf'),pagesize=(960,540));c.setTitle('Потребительские профили муниципалитетов');c.setAuthor('')
    dark=HexColor('#153748');muted=HexColor('#55717f')
    def text(s,x,y,width=830,size=18,color=dark,bold=False):
        st=ParagraphStyle('slide',fontName='DV-Bold' if bold else 'DV',fontSize=size,leading=size*1.35,textColor=color)
        p=Paragraph(s,st);_,h=p.wrap(width,500);p.drawOn(c,x,y-h);return h
    page=0
    def start(title,subtitle=''):
        nonlocal page
        page+=1;c.setFillColor(white);c.rect(0,0,960,540,fill=1,stroke=0)
        text(title,42,500,size=29,bold=True)
        if subtitle:text(subtitle,44,450,size=12,color=muted)
        c.setStrokeColor(HexColor('#d8e3e6'));c.line(42,43,918,43)
        text('СберИндекс · 2023–2024',42,30,size=9,color=muted)
        text(str(page),880,30,width=35,size=9,color=muted)
    def fig(name,x,y,width,height):c.drawImage(str(REPORTS/'figures'/(name+'.png')),x,y,width,height,preserveAspectRatio=True,anchor='c',mask='auto')
    start('Потребительские профили<br/>муниципалитетов')
    text('2 016',44,360,size=58,bold=True);text('муниципалитетов',46,278,size=18)
    text('24 месяца · 5 типов',46,219,size=24,bold=True)
    text('Сходство корзины расходов<br/>и переходы между типами',46,169,width=350,size=18,color=muted)
    fig('network',410,97,510,310);c.showPage()
    start('Данные и архитектура исследования','Снимок расходов СберИндекса за 2023–2024 годы')
    steps=[('01','Данные','2 190 идентификаторов; 174 неполных ряда исключены.'),('02','Атрибуты','CLR корзины и относительный уровень расходов.'),('03','Сеть','20 ближайших соседей; локальный масштаб весов.'),('04','Типология','Компактность + связи + устойчивость во времени.'),('05','Проверка','Четыре метода, внутренние метрики и внешние показатели.')]
    for i,(num,h,b) in enumerate(steps):
        y=410-i*62;text(num,45,y,size=23,color=HexColor('#249687'),bold=True);text(h,109,y,width=210,size=19,bold=True);text(b,330,y,width=575,size=16)
    c.showPage()
    start('Пять профилей потребительского спроса','Медианы долей в декабре 2024; сумма медиан может отличаться от 100%')
    fig('profiles',42,108,876,310)
    text('Т1: базовый спрос · Т2: смешанный · Т3: низкая доля транспорта<br/>Т4: высокие расходы при удалённости рынков · Т5: спрос на услуги',48,98,size=13)
    c.showPage()
    start('Сравнение качества и динамики','Средние метрики за июль 2023 – декабрь 2024; K = 5 у всех методов')
    r=t['method_comparison'].set_index('method')
    cols=['SW','S_Dbw','AVI','AVU','MQ','ARI_previous'];xs=[43,235,345,455,565,675,785]
    headers=['Метод','SW ↑','S_Dbw ↓','AVI ↑','AVU ↓','MQ ↑','ARI во времени ↑']
    for x,h in zip(xs,headers):text(h,x,395,width=145,size=13,bold=True)
    names={'graph_temporal':'Сеть + время','kmeans':'K-means','spectral':'Спектральная','ward':'Ward'}
    for j,(key,label) in enumerate(names.items()):
        y=347-j*48;text(label,xs[0],y,width=185,size=17,bold=key=='graph_temporal')
        for x,col in zip(xs[1:],cols):text(f'{r.loc[key,col]:.3f}',x,y,width=100,size=17)
    text('У K-means чуть выше силуэт. Временная устойчивость основной модели<br/>частично задана штрафом за смену группы.',44,131,size=18,color=muted)
    c.showPage()
    start('Внешний экономический контекст','Внешние показатели не участвуют в обучении')
    fig('external',36,157,884,255)
    text('Тип 4: высокая зарплата и низкая доступность рынков.<br/>Тип 5: высокая зарплата, более крупные рынки и больше расходов на услуги.',48,137,size=18)
    text('Медианы по доступным наблюдениям. Причинная связь не устанавливается.',48,71,size=11,color=muted)
    c.showPage()
    start('Траектории муниципалитетов','Одна строка — одно МО. Цвет — тип. Порядок строк задан декабрем 2024.')
    fig('trajectories',50,109,860,307)
    text('Средняя доля переходов: 9,50% за месяц.<br/>Без временного штрафа: 16,56% на том же оценочном периоде.',50,101,size=16)
    c.showPage()
    start('Устойчивость и реальные примеры','Десять подвыборок по 80% МО; временной штраф в этой проверке выключен')
    fig('stability',37,169,534,241)
    text('Средний ARI<br/><b>0,866</b> · сеть + время<br/><b>0,908</b> · K-means',601,390,width=295,size=20)
    text('Богородский район, Кировская область: тип 2 → 1.<br/>Ульчский район, Хабаровский край: тип 2 → 1.<br/>Катангский район: сильное изменение атрибутов при сохранении типа 4.',48,143,size=16)
    c.showPage()
    start('Использование и ограничения')
    text('Подбор сопоставимых муниципалитетов',44,403,size=22,bold=True)
    text('Тип дает отправную точку для анализа спроса. Затем нужно проверить<br/>исходные атрибуты территории и местный экономический контекст.',44,358,size=18)
    text('Границы вывода',44,281,size=22,bold=True)
    text('Два года наблюдений. Сезонность, цены, охват платежей и изменения<br/>границ могут влиять на результаты. Переход между группами<br/>не доказывает структурный шок.',44,237,size=18)
    text('Данные, код, конфигурация и расчетные таблицы вложены в репозиторий.<br/>Для просмотра отдельного МО: explorer.html.',44,132,size=16,color=muted)
    text('Источник: СберИндекс, публичная копия набора Data → Sense. Адреса и SHA-256: data/sources.json.',44,77,size=10,color=muted)
    c.showPage();c.save()


if __name__=='__main__':
    REPORTS.mkdir(exist_ok=True)
    data=load(); plots(data); results_text(data); pdf_reports(data)
    print('PDF, графики и интерактивный просмотр созданы.')
