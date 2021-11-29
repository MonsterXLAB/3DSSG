import sys
sys.path.append("./src")
from src.config import Config
from src.SceneGraphFusionNetwork import SGFN
import torch


# def visualize(data_dict, model, obj_dict, pred_dict):
#     torch.backends.cudnn.enabled = False
#     dot = Digraph(comment='The Scene Graph')
#     dot.attr(rankdir='TB')

#     with torch.no_grad():
#         data_dict = model(data_dict)

#     data, pred_relations = get_eval(data_dict)
#     triples = data["triples"][0].cpu().numpy()
#     object_id = data["objects_id"][0].cpu().numpy()
#     object_cat = data["objects_cat"][0].cpu().numpy()
#     object_pred = data["objects_predict"][0].cpu().numpy()

#     dot.attr(label='predicted')
#     # nodes
#     obj_pred_cls = np.argmax(object_pred, axis=1)
#     dot.attr('node', shape='oval', fontname='Sans')
#     for index in range(len(object_cat)):
#         id = str(object_id[index])
#         dot.attr('node', fillcolor=node_color_list[index], style='filled')
#         pred = obj_pred_cls[index]
#         gt = object_cat[index]
#         note = obj_dict[pred] + '\n(GT:' + obj_dict[gt] + ')'
#         dot.node(id, note)
#     # edges
#     dot.attr('edge', fontname='Sans', color='black', style='filled')
#     for relation in pred_relations[0][:50]:
#         s, o, p = relation
#         if p == 0:
#             continue
#         dot.edge(str(s), str(o), pred_dict[p])
#     for item in triples:
#         s, o, p = item
#         if p == 0:
#             continue
#         dot.edge(str(s), str(o), pred_dict[p])

#     # print(dot.source)
#     scan = data_dict["scan_id"][0][:-2]
#     split = data_dict["scan_id"][0][-1]
#     dot.render(filename=os.path.join(
#         CONF.PATH.BASE, 'vis/{}/scene_graph_{}.gv'.format(scan, split)))


def main():
    config = Config("config_CVPR21.json")
    config.MODE = 'eval'
    config.PATH = '/3DSSG/'
    config.NAME = 'CVPR21'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config.DEVICE = device
    model = SGFN(config=config)
    print(">>>> Loading model with RGB=False, Normal=False")
    model.load()
    model.eval(debug_mode=False)

if __name__ == "__main__":
    main()