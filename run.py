import fire
from dependency_injector import containers, providers
from line_segmentation.PreciseLineSegmenter import PreciseLineSegmenter
from line_segmentation.HBLineSegmenter import HBLineSegmenter
from transcriber.wrappers.visionED_wrapper import VisionEncoderDecoderTranscriber
from transcriber.wrappers.hybrid_wrapper import HybridTranscriber
from transcriber.wrappers.vit_wrapper import VitTranscriber
from xmlgenerator import generate_xml
import os
from PIL import Image
import numpy as np
import time

class LineSegmenterContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    selector = providers.Selector(
        config.segmenter,
        precise=providers.Factory(PreciseLineSegmenter),
        height_based=providers.Factory(HBLineSegmenter),
    )

class TranscriberContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    selector = providers.Selector(
        config.transcriber,
        visioned=providers.Factory(VisionEncoderDecoderTranscriber),
        hybrid=providers.Factory(HybridTranscriber),
        vit=providers.Factory(VitTranscriber),
    )
class NHMDPipeline(object):
    def __init__(self, config_path='./pipeline_config.json', save_images=False, out_dir='./out', out_type='txt'):
        os.makedirs(out_dir, exist_ok=True)
        if save_images:
            os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)

        self.out_dir = out_dir
        self.out_type = out_type
        self.save_images = save_images

        lscontainer = LineSegmenterContainer()
        lscontainer.config.from_json(config_path)
        self.segmenter = lscontainer.selector()

        tcontainer = TranscriberContainer()
        tcontainer.config.from_json(config_path)
        self.transcriber = tcontainer.selector()

    def evaluate_baseline(self, path):
        _, _, clusters, _, _ = self.segmenter.segment_lines(path)
        baselines = ''
        for i in range(1, len(clusters)):
            Si = clusters[i]
            for p in Si.keys():
                pointx = p[1]
                pointy = p[0]
                baselines += f'{pointx},{pointy};'
            baselines += '\n'
        return baselines

    def evaluate_baselines(self, path):
        print('Started evaluation.')
        start = time.time()
        out_dir = './preds'
        os.makedirs(out_dir, exist_ok=True)
        for file in os.listdir(path):
            if file.endwith('.jpg') or file.endwith('.png'):
                baselines = self.evaluate_baseline(os.path.join(path, file), )
                with open(os.path.join(out_dir, file[:-4] + '.txt'), 'w') as f:
                    f.write(baselines)
        end = time.time()
        print(f"End of processing. Runtime: {int(end-start)} seconds")

    def process_image(self, path, id=None):
        if id is None:
            print('Starting processing...')
            start = time.time()

        lines, polygons, baselines, region_coords, scale = self.segmenter.segment_lines(path)
        predictions = []
        for idx, line in enumerate(lines):
            text_line = f'{id}_line_{idx}.jpg' if id is not None else f'line_{idx}.jpg'
            if self.save_images:
                img = Image.fromarray(line*255).convert('L')
                img.save(os.path.join(self.out_dir, 'images', text_line))
            pred = self.transcriber.transcribe(np.array(line*255))
            predictions.append({'file':text_line, 'pred':pred})

        if self.out_type == 'txt':
            txt_predictions = [f'{pred["file"]}\t{pred["pred"]}\n'for pred in predictions]
            with open(os.path.join(self.out_dir, 'result.txt'), 'w') as f:
                f.write(''.join(txt_predictions))
        elif self.out_type == 'xml':
            filename = path.split('/')[-1]
            transcriptions = [d["pred"] for d in predictions]
            generate_xml(filename, polygons, baselines,
                         region_coords, scale, transcriptions, self.out_dir)
        if id is None:
            end = time.time()
            print(f"End of processing. Inference time: {int(end-start)} seconds")

    def process_dir(self, path):
        print('Starting processing...')
        start = time.time()
        for file in os.listdir(path):
            if file.endwith('.jpg') or file.endwith('.png'):
                self.process_image(os.path.join(path, file), file[:-4])
        end = time.time()
        print(f"End of processing. Inference time: {int(end-start)} seconds")

if __name__ == '__main__':
    # "./line_segmentation/demo/orig.jpg"
    fire.Fire(NHMDPipeline)