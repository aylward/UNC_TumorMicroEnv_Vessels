import os

import numpy as np

import itk
from itk import TubeTK as tube

class usangio_vessels:
    def __init__(self, debug=False):
        self.debug = debug

        self.output_base_name = []
        
        self.norm_blur = 1.5
        self.intensity_z_score_range = 10

        self.num_training_seeds = 25
        self.num_seeds = 1000
        self.min_vessel_length = 500

        self.data_im = []
        self.data_norm_array = []
        self.data_norm_im = []

        self.training_seed_coord = []
        self.training_mask_image = []

        self.vessels_enhancement_scales = [1, 2, 4]

        self.data_vessels_enhanced_im = []
        self.data_vessels_im = []
        self.data_vessels_group = []

    def read_image_file(self, filename):
        print("Reading image")
        
        self.data_im = itk.imread(filename, itk.F)
        self.output_base_name = []
        
        fname = os.path.basename(filename)
        fbasename = os.path.splitext(fname)[0]
        dname = os.path.join(".", "Results")
        if not os.path.exists(dname):
            os.makedirs(dname)
        self.output_base_name = os.path.join(dname, fbasename+"-out")

    def read_matlab_file(self, filename, nx=460, ny=308, nz=510, bits=32):
        print("Reading MATLAB file")
        
        with open(filename, mode="rb") as file:
            file_content = file.read()
        pixel_type = None
        if bits == 32:
            pixel_type = np.dtype(np.float32)
        elif bits == 64:
            pixel_type = np.dtype(np.float64)
        elif bits == 16:
            pixel_type = np.dtype(np.float16)
        else:
            print("ERROR: Only 16, 32, and 64 bit floats supported")
            self.data_im = []
            return

        data_1d = np.frombuffer(file_content)
        data_raw = np.reshape(data_1d, [nx, ny, nz], order="C")
        self.data_im = itk.GetImageFromArray(data_raw.as_type(np.float32))
        
        fname = os.path.basename(filename)
        fbasename = os.path.splitext(fname)[0]
        dname = os.path.join(".", "Results")
        if not os.path.exists(dname):
            os.makedirs(dname)
        self.output_base_name = os.path.join(dname, fbasename+"-out")

    def import_matlab_data(self, data):
        print("Importing MATLAB data")
        
        self.data_im = itk.GetImageFromArray(data.as_type(np.float32))
        
        dname = os.path.join(".", "Results")
        if not os.path.exists(dname):
            os.makedirs(dname)
        self.output_base_name = os.path.join(dname, "matlab-out")

    def normalize_image(self):
        print("Normalizing image")
        
        imMath = tube.ImageMath.New(self.data_im)
        imMath.Blur(self.norm_blur)
        imBlur = imMath.GetOutput()

        data_norm = itk.GetArrayFromImage(imBlur)
        data_mean = np.mean(data_norm)
        data_stddev = np.std(data_norm)

        data_norm = (data_norm - data_mean) / (
            data_stddev * self.intensity_z_score_range
        )
        data_norm = np.where(data_norm > 1, 1.0, data_norm)
        data_norm = np.where(data_norm < -1, -1.0, data_norm)
        data_min = data_norm.min()
        data_max = data_norm.max()
        self.data_norm_array = (data_norm - data_min) / (data_max - data_min) * 100
        self.data_norm_im = itk.GetImageFromArray(data_norm)
        itk.imwrite(
            self.data_norm_im,
            self.output_base_name + "-Normalized.mha"
        )

    def identify_training_seeds(self):
        print("Identifying training seeds")
        
        seed_coverage = 20
        data_min = self.data_norm_array.min()
        self.training_seed_coord = np.zeros([self.num_training_seeds, 3])
        for i in range(self.num_training_seeds):
            self.training_seed_coord[i] = np.unravel_index(
                np.argmax(self.data_norm_array, axis=None), self.data_norm_array.shape
            )
            indx = [
                int(self.training_seed_coord[i][0]),
                int(self.training_seed_coord[i][1]),
                int(self.training_seed_coord[i][2]),
            ]
            minX = max(indx[0] - seed_coverage, 0)
            maxX = min(indx[0] + seed_coverage, self.data_norm_array.shape[0])
            minY = max(indx[1] - seed_coverage, 0)
            maxY = min(indx[1] + seed_coverage, self.data_norm_array.shape[1])
            minZ = max(indx[2] - seed_coverage, 0)
            maxZ = min(indx[2] + seed_coverage, self.data_norm_array.shape[2])
            self.data_norm_array[minX:maxX, minY:maxY, minZ:maxZ] = data_min
            indx.reverse()
            self.training_seed_coord[:][
                i
            ] = self.data_norm_im.TransformIndexToPhysicalPoint(indx)

    def generate_training_mask_image(self):
        print("Generating training mask")
        
        # Manually extract a few vessels to form an
        #     image-specific training set
        ImageType = itk.Image[itk.F, 3]
        LabelMapType = itk.Image[itk.UC, 3]

        print("   - Extracting training vessels")
        vSeg = tube.SegmentTubes.New(Input=self.data_norm_im)
        vSeg.SetVerbose(True)
        vSeg.SetMinRoundness(0.1)
        vSeg.SetMinRidgeness(0.8)
        vSeg.SetMinCurvature(0.000001)
        # MinCurvature is the most influential variable
        vSeg.SetRadiusInObjectSpace(1.5)
        vSeg.SetMinLength(300)
        for i in range(self.num_training_seeds):
            print("     Seed", i, "of", self.num_training_seeds)
            vSeg.ExtractTubeInObjectSpace(self.training_seed_coord[i], i)
        self.vessel_mask_image = vSeg.GetTubeMaskImage()

        print("   - Computing training mask (This takes several minutes)")
        print("     ...")
        trMask = tube.ComputeTrainingMask[ImageType, LabelMapType].New()
        trMask.SetInput(self.vessel_mask_image)
        trMask.SetGap(10)
        trMask.SetObjectWidth(1)
        trMask.SetNotObjectWidth(1)
        trMask.Update()
        self.vessel_mask_image = trMask.GetOutput()
        itk.imwrite(
            self.vessel_mask_image,
            self.output_base_name + "-VesselsTrainingMask.mha"
        )

    def enhance_vessels(self):
        print("Enhancing vessels (This takes several minutes)")
        print("...")
        ImageType = itk.Image[itk.F, 3]
        LabelMapType = itk.Image[itk.UC, 3]

        enhancer = tube.EnhanceTubesUsingDiscriminantAnalysis[
            ImageType, LabelMapType
        ].New()
        enhancer.AddInput(self.data_norm_im)
        enhancer.SetLabelMap(self.vessel_mask_image)
        enhancer.SetRidgeId(255)
        enhancer.SetBackgroundId(128)
        enhancer.SetUnknownId(0)
        enhancer.SetTrainClassifier(True)
        enhancer.SetUseIntensityOnly(True)
        enhancer.SetUseFeatureMath(True)
        enhancer.SetScales(self.vessels_enhancement_scales)
        enhancer.Update()
        enhancer.ClassifyImages()

        imMath = tube.ImageMath.New(enhancer.GetClassProbabilityImage(0))
        imMath.Blur(0.5)
        prob0 = imMath.GetOutput()
        imMath.SetInput(enhancer.GetClassProbabilityImage(1))
        imMath.Blur(0.5)
        prob1 = imMath.GetOutput()

        imDiff = itk.SubtractImageFilter(Input1=prob0, Input2=prob1)
        imDiffArr = itk.GetArrayFromImage(imDiff)
        dMax = imDiffArr.max()
        imProbArr = imDiffArr / dMax
        self.data_vessels_enhanced_im = itk.GetImageFromArray(imProbArr)
        self.data_vessels_enhanced_im.CopyInformation(self.data_im)
        itk.imwrite(
            self.data_vessels_enhanced_im,
            self.output_base_name + "-VesselsEnhanced.mha"
        )

    def extract_vessels_from_enhanced_image(self):
        print("Extract vessels (This takes several minutes)")
        imMath = tube.ImageMath.New(self.data_vessels_enhanced_im)
        imMath.MedianFilter(1)
        imMath.Threshold(0.000001, 1, 1, 0)
        imVessMask = imMath.GetOutputShort()

        ccSeg = tube.SegmentConnectedComponents.New(imVessMask)
        ccSeg.SetMinimumVolume(300)
        ccSeg.Update()
        imVessMask = ccSeg.GetOutput()
        itk.imwrite(imVessMask, self.output_base_name + "-VesselSeedsInitialMask.mha")
        imVessMask = itk.imread(self.output_base_name + "-VesselSeedsInitialMask.mha", itk.F)
        imMath.SetInput(self.data_vessels_enhanced_im)
        imMath.ReplaceValuesOutsideMaskRange(imVessMask, 1, 99999, 0)
        imSeeds = imMath.GetOutput()
        itk.imwrite(imSeeds, self.output_base_name + "-VesselSeeds.mha")

        print("   - Extracting seed info")
        imMath.SetInput(imVessMask)
        imMath.Threshold(0, 1, 1, 0)
        imVessMaskInv = imMath.GetOutput()
        distFilter = itk.DanielssonDistanceMapImageFilter.New(imVessMaskInv)
        distFilter.Update()
        dist = distFilter.GetOutput()

        imMath.SetInput(dist)
        imMath.Blur(0.4)
        tmp = imMath.GetOutput()
        imMath.ReplaceValuesOutsideMaskRange(tmp, 0.1, 10, 0)
        imSeedsRadius = imMath.GetOutput()
        itk.imwrite(imSeedsRadius, self.output_base_name + "-VesselSeedsRadius.mha")

        imMath.SetInput(self.data_norm_im)
        imMath.ReplaceValuesOutsideMaskRange(imVessMask, 1, 99999, 0)
        imMath.Blur(1)
        imInput = imMath.GetOutput()
        itk.imwrite(imInput, self.output_base_name + "-VesselInput.mha")

        print("   - Extracting vessels")
        vSeg = tube.SegmentTubes.New(Input=imInput)
        vSeg.SetVerbose(True)
        vSeg.SetMinCurvature(0.001)
        vSeg.SetMinRoundness(0.1)
        vSeg.SetMinRidgeness(0.1)
        vSeg.SetMinLevelness(0.001)
        vSeg.SetMinLength(self.min_vessel_length)
        # vSeg.SetRadiusInObjectSpace( 1.5 )
        vSeg.SetBorderInIndexSpace(3)
        vSeg.SetSeedMask(imSeeds)
        vSeg.SetSeedRadiusMask(imSeedsRadius)
        vSeg.SetOptimizeRadius(True)
        vSeg.SetSeedMaskMaximumNumberOfPoints(self.num_seeds)
        vSeg.SetUseSeedMaskAsProbabilities(True)
        vSeg.SetSeedExtractionMinimumProbability(0.1)
        vSeg.ProcessSeeds()
        self.data_vessels_im = vSeg.GetTubeMaskImage()

        print("   - Saving results")
        itk.imwrite(
            self.data_vessels_im, self.output_base_name + "-Vessels.mha"
        )

        TubeMath = tube.TubeMath[3, itk.F].New()
        TubeMath.SetInputTubeGroup(vSeg.GetTubeGroup())
        TubeMath.SetUseAllTubes()
        TubeMath.SmoothTube(4, "SMOOTH_TUBE_USING_INDEX_GAUSSIAN")
        TubeMath.SmoothTubeProperty("Radius", 2, "SMOOTH_TUBE_USING_INDEX_GAUSSIAN")
        self.data_vessels_group = TubeMath.GetOutputTubeGroup()

        SOWriter = itk.SpatialObjectWriter[3].New()
        SOWriter.SetInput(self.data_vessels_group)
        SOWriter.SetBinaryPoints(True)
        SOWriter.SetFileName(self.output_base_name + "-Vessels.tre")
        SOWriter.Update()

        ConvSurface = tube.WriteTubesAsPolyData.New()
        ConvSurface.SetInput(self.data_vessels_group)
        ConvSurface.SetFileName(self.output_base_name + "-Vessels.vtp")
        ConvSurface.Update()

    def run(self):
        self.normalize_image()
        self.identify_training_seeds()
        self.generate_training_mask_image()
        self.enhance_vessels()
        self.extract_vessels_from_enhanced_image()
