#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

/*
#if __has_cpp_attribute(__cpp_lib_math_special_functions)
#define CYL_BESSEL_I(nu, x) std::cyl_bessel_i(nu, x)
#else
#include <boost/math/special_functions/bessel.hpp>
#define CYL_BESSEL_I(nu, x) boost::math::cyl_bessel_i(nu, x)
#endif
*/

#define CYL_BESSEL_I(nu, x) std::cyl_bessel_i(nu, x)

using namespace cv;

std::vector<float> kaiser(int M, float beta) {
    if (M <= 0) return {};
    std::vector<float> ret(M);
    for (int i = 0; i < M; ++i) {
        auto n = i - float(M - 1) / 2;
        auto a = beta * std::sqrt(1.f - float(4 * n * n) / ((M - 1) * (M - 1)));
        ret[i] = CYL_BESSEL_I(0, a) / CYL_BESSEL_I(0, beta);
    }
    return ret;
}

Mat kaiser2DSquare(int M, float beta) {
    auto temp = kaiser(M, beta);
    Mat a(temp);
    Mat a_T;
    transpose(a, a_T);
    return a * a_T;
}

void test() {
    float c[]{1, 1, 2, 3};
    Mat a(2, 2, CV_32F, c);
    std::cout << a;
    // a=a.mul(a);
    // a /= a;
    a *= 2;
    std::cout << a;
}

void addNoise(Mat img) {
    // gaussian noise
    Mat noise(img.size(), CV_8S);
    randn(noise, 0, 20);
    // std::cout << noise;
    img += noise;
}

void createNoisyImage() {
    std::string image_path = samples::findFile("lenna.png");
    Mat img = imread(image_path);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return;
    }
    // convert to gray
    cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);

    // split channgel
    addNoise(img);
    imwrite("output_noisy.png", img);
}

//================================bm3d=======================

float bm3d_sigma = 20;

// step 1 parameters
struct ParamStep1 {
    // ht means hard thresholding
    int N_1_ht = 6;        // block size
    int N_2_ht = 16;        // max block num
    int N_step_ht = 4;      // search step
    int N_S_ht = 39;        // search window size
    int N_FS_ht = 1;        //
    int N_PR_ht = 0;        //
    float beta_ht = 2.0;    //
    float lambda_2d = 2.0;  //
    float lambda_3d = 2.8;  //
    float threshold_ht = 300;

    Mat kaiserWeight;
} param_step_1{};

// step2 parameters
struct ParamStep2 {
    // wie means Wiener filtering
    int N_1_wie = 6;      // block size
    int N_2_wie = 16;      //
    int N_step_wie = 6;    //
    int N_S_wie = 39;      //
    int N_FS_wie = 1;      //
    int N_PR_wie = 0;      //
    float beta_wie = 2.0;  //
    float threshold_wie = 100;

    Mat kaiserWeight;
} param_step_2{};

struct BlockInfo {
    float depth_to_ref;
    int x;
    int y;
    Mat data;
};
struct BlockGroup {
    Mat image;
    int blockWidth;
    int blockHeight;
    std::vector<BlockInfo> blocks;
};

float DSNR(Mat img1, Mat img2) {
    assert(img1.size() == img2.size());

    Mat temp1, temp2;
    img1.convertTo(temp1, CV_32F);
    img2.convertTo(temp2, CV_32F);

    Mat m = temp1 - temp2;
    m = m.mul(m);

    return 20 * log10(255. / sqrt(sum(m)[0] / (m.rows * m.cols)));
}

float step_1_block_distance(Mat ref_block_dct, Mat block_dct) {
    auto ext = ref_block_dct.size();
    auto th = param_step_1.lambda_2d * bm3d_sigma;

    Mat temp1, temp2;
    ref_block_dct.copyTo(temp1);
    block_dct.copyTo(temp2);

    temp1.forEach<float>([th](auto& e, auto&&) {
        if (e < th) e = 0;
    });
    temp2.forEach<float>([th](auto& e, auto&&) {
        if (e < th) e = 0;
    });
    auto temp = temp1 - temp2;
    temp = temp.mul(temp);
    return float(sum(temp)[0]) / (ext.width * ext.height);
}

Mat block_dct(Mat image, Rect rect) {
    auto ext = image.size();
    int padx = std::max(rect.x + rect.width - ext.width, 0);
    int pady = std::max(rect.y + rect.height - ext.height, 0);
    Mat ret;
    if (padx || pady) {
        Mat m;
        copyMakeBorder(image(Rect{rect.x, rect.y, rect.width - padx, rect.height - pady}), m, 0,
                       pady, 0, padx, BORDER_CONSTANT);
        dct(m, ret);
    } else
        dct(image(rect), ret);
    return ret;
}

BlockGroup step_1_block_matching(Mat image, int refx, int refy) {
    auto ext = image.size();

    int blockWidthLimit, blockHeightLimit;
    blockWidthLimit = std::min(param_step_1.N_1_ht, ext.width - refx);
    blockHeightLimit = std::min(param_step_1.N_1_ht, ext.height - refy);

    Mat refBlockDct = block_dct(image, Rect{refx, refy, param_step_1.N_1_ht, param_step_1.N_1_ht});

    Vec4i searchWindow;
    searchWindow[0] = std::max(0, refx - param_step_1.N_S_ht);
    searchWindow[1] = std::max(0, refy - param_step_1.N_S_ht);
    searchWindow[2] = std::min(ext.width, refx + param_step_1.N_S_ht);
    searchWindow[3] = std::min(ext.height, refy + param_step_1.N_S_ht);

    BlockGroup ret{image, param_step_1.N_1_ht, param_step_1.N_1_ht};
    ret.blocks.emplace_back(BlockInfo{0, refx, refy, refBlockDct});

    for (int j = searchWindow[1]; j < searchWindow[3] - blockHeightLimit;
         j += param_step_1.N_step_ht) {
        for (int i = searchWindow[0]; i < searchWindow[2] - blockWidthLimit;
             i += param_step_1.N_step_ht) {
            if (i == refx && j == refy) continue;
            Mat blockDct = block_dct(image, Rect{i, j, param_step_1.N_1_ht, param_step_1.N_1_ht});
            auto block_d = step_1_block_distance(refBlockDct, blockDct);
            if (block_d < param_step_1.threshold_ht) {
                ret.blocks.emplace_back(BlockInfo{block_d, i, j, blockDct});
            }
        }
    }
    std::sort(ret.blocks.begin(), ret.blocks.end(),
              [](auto&& a, auto&& b) { return a.depth_to_ref < b.depth_to_ref; });

    auto count = ret.blocks.size();
    if (count > param_step_1.N_2_ht) ret.blocks.resize(param_step_1.N_2_ht);
    return ret;
}

void step_1_filter_3d(BlockGroup& blockGroup) {
    int nonZeroCount{};
    int nz = blockGroup.blocks.size();
    int nx = blockGroup.blockWidth;
    int ny = blockGroup.blockHeight;

    bool flag = nz % 2;

    auto th = param_step_1.lambda_3d * bm3d_sigma;

    // do 3d transform
    if (nz > 1) {
        for (int x = 0; x < nx; ++x)
            for (int y = 0; y < ny; ++y) {
                std::vector<float> tempDct(flag ? nz + 1 : nz);
                for (int z = 0; z < nz; ++z) {
                    tempDct[z] = blockGroup.blocks[z].data.at<float>(y, x);
                }
                dct(tempDct, tempDct);

                for (auto& e : tempDct) {
                    if (abs(e) < th)
                        e = 0;
                    else
                        ++nonZeroCount;
                }

                idct(tempDct, tempDct);
                for (int z = 0; z < nz; ++z) {
                    blockGroup.blocks[z].data.at<float>(y, x) = tempDct[z];
                }
            }
    }

    //
    Mat weight;
    param_step_1.kaiserWeight.copyTo(weight);
    // weight = Mat::ones(param_step_1.N_1_ht, param_step_1.N_1_ht, CV_32F);
    if (nonZeroCount > 1) weight /= bm3d_sigma * bm3d_sigma * nonZeroCount;

    Mat weightSum = Mat::zeros(weight.size(), weight.type());
    for (int i = 0; i < nz; ++i) {
        idct(blockGroup.blocks[i].data, blockGroup.blocks[i].data);

        blockGroup.blocks[i].data = blockGroup.blocks[i].data.mul(weight);
        weightSum += weight;
        if (i > 0) blockGroup.blocks[0].data += blockGroup.blocks[i].data;
    }
    // std::cout << "weightSum " << weightSum << std::endl;
    blockGroup.blocks[0].data /= weightSum;
}
void bm3d_aggregation(Mat img, std::vector<BlockGroup>& blockGroups) {
    auto ext = img.size();
    for (auto&& e : blockGroups) {
        auto rect =
            Rect{e.blocks[0].x, e.blocks[0].y, std::min(e.blockWidth, ext.width - e.blocks[0].x),
                 std::min(e.blockHeight, ext.height - e.blocks[0].y)};
        e.blocks[0].data(Rect{0, 0, rect.width, rect.height}).copyTo(img(rect));
    }
}
Mat bm3d_step1(Mat noisyImage) {
    std::vector<BlockGroup> blockGroups;
    auto ext = noisyImage.size();

    for (int j = 0; j < ext.height; j += param_step_1.N_1_ht) {
        for (int i = 0; i < ext.width; i += param_step_1.N_1_ht) {
            // auto p = noisyImage->getDataPtr(i, j);
            blockGroups.emplace_back(step_1_block_matching(noisyImage, i, j));
        }
    }

    for (int i = 0; i < blockGroups.size(); ++i) {
        // std::cout << "block count: " << blockGroups[i].blocks.size() << std::endl;
        step_1_filter_3d(blockGroups[i]);
    }
    Mat ret(noisyImage.size(), noisyImage.type());
    bm3d_aggregation(ret, blockGroups);
    return ret;
}
//=======================================================
float step_2_block_distance(Mat ref_block, Mat block) {
    auto ext = ref_block.size();
    auto temp = ref_block - block;
    temp = temp.mul(temp);
    return float(sum(temp)[0]) / (ext.width * ext.height);
}
Mat step_2_block(Mat image, Rect rect) {
    auto ext = image.size();
    int padx = std::max(rect.x + rect.width - ext.width, 0);
    int pady = std::max(rect.y + rect.height - ext.height, 0);
    Mat ret;
    if (padx || pady) {
        copyMakeBorder(image(Rect{rect.x, rect.y, rect.width - padx, rect.height - pady}), ret, 0,
                       pady, 0, padx, BORDER_CONSTANT);
    } else
        ret = image(rect).clone();
    return ret;
}
void step_2_block_matching(BlockGroup& blockGroup, Mat image, int refx, int refy) {
    blockGroup.image = image;
    blockGroup.blockWidth = param_step_2.N_1_wie;
    blockGroup.blockHeight = param_step_2.N_1_wie;

    auto ext = image.size();

    int blockWidthLimit, blockHeightLimit;
    blockWidthLimit = std::min(param_step_2.N_1_wie, ext.width - refx);
    blockHeightLimit = std::min(param_step_2.N_1_wie, ext.height - refy);

    Mat refBlock =
        step_2_block(image, Rect{refx, refy, param_step_2.N_1_wie, param_step_2.N_1_wie});

    Vec4i searchWindow;
    searchWindow[0] = std::max(0, refx - param_step_2.N_S_wie);
    searchWindow[1] = std::max(0, refy - param_step_2.N_S_wie);
    searchWindow[2] = std::min(ext.width, refx + param_step_2.N_S_wie);
    searchWindow[3] = std::min(ext.height, refy + param_step_2.N_S_wie);

    blockGroup.blocks.emplace_back(BlockInfo{0, refx, refy, refBlock});

    for (int j = searchWindow[1]; j < searchWindow[3] - blockHeightLimit;
         j += param_step_2.N_step_wie) {
        for (int i = searchWindow[0]; i < searchWindow[2] - blockWidthLimit;
             i += param_step_2.N_step_wie) {
            if (i == refx && j == refy) continue;
            Mat block = step_2_block(image, Rect{i, j, param_step_2.N_1_wie, param_step_2.N_1_wie});
            auto block_d = step_2_block_distance(refBlock, block);
            if (block_d < param_step_2.threshold_wie) {
                blockGroup.blocks.emplace_back(BlockInfo{block_d, i, j, block});
            }
        }
    }
    std::sort(blockGroup.blocks.begin(), blockGroup.blocks.end(),
              [](auto&& a, auto&& b) { return a.depth_to_ref < b.depth_to_ref; });

    auto count = blockGroup.blocks.size();
    if (count > param_step_2.N_2_wie) blockGroup.blocks.resize(param_step_2.N_2_wie);
}
void step_2_filter_3d(std::vector<BlockGroup>& noisyBlockGroups,
                      std::vector<BlockGroup>& basicBlockGroups) {
    auto groupCount = noisyBlockGroups.size();
    auto blockWidth = param_step_2.N_1_wie;
    auto blockHeight = param_step_2.N_1_wie;

    // do 3d transform for basic group
    for (int u = 0; u < groupCount; ++u) {
        auto&& basicGroup = basicBlockGroups[u];
        auto&& noisyGroup = noisyBlockGroups[u];
        int blockCount = basicGroup.blocks.size();
        // std::cout << "step2 blockCount: " << blockCount << std::endl;

        float weight = 0;
        float wienerWeight = 1;
        // 2d dct
        for (auto&& block : basicGroup.blocks) dct(block.data, block.data);
        for (auto&& block : noisyGroup.blocks) dct(block.data, block.data);

        // 3d dct
        for (int j = 0; j < blockHeight; ++j) {
            for (int i = 0; i < blockWidth; ++i) {
                // basic block
                auto tempBlockCount = blockCount % 2 ? (blockCount + 1) : blockCount;
                Mat temp(tempBlockCount, 1, CV_32F);
                for (int k = 0; k < blockCount; ++k) {
                    temp.at<float>(k, 0) = basicGroup.blocks[k].data.at<float>(j, i);
                }
                dct(temp, temp);
                temp = temp.mul(temp);
                temp /= temp.rows;
                temp /= (temp + bm3d_sigma * bm3d_sigma);

                weight += sum(temp)[0];

                // noisy block
                Mat temp2(tempBlockCount, 1, CV_32F);
                for (int k = 0; k < blockCount; ++k)
                    temp2.at<float>(k, 0) = noisyGroup.blocks[k].data.at<float>(j, i);
                dct(temp2, temp2);
                temp2 = temp2.mul(temp);

                idct(temp2, temp2);

                for (int k = 0; k < blockCount; ++k)
                    noisyGroup.blocks[k].data.at<float>(j, i) = temp2.at<float>(k, 0);
            }
        }
        if (weight > 0) wienerWeight = 1. / (weight * bm3d_sigma * bm3d_sigma);

        Mat blockWeight = param_step_2.kaiserWeight.clone();
        // Mat blockWeight = Mat::ones(param_step_2.N_1_wie, param_step_2.N_1_wie, CV_32F);
        blockWeight *= wienerWeight;

        Mat weightSum = Mat::zeros(param_step_2.N_1_wie, param_step_2.N_1_wie, CV_32F);
        for (int i = 0; i < blockCount; ++i) {
            auto m = noisyGroup.blocks[i].data;
            idct(m, m);
            m = blockWeight.mul(m);
            if (i > 0) noisyGroup.blocks[0].data += m;
            weightSum += blockWeight;
        }
        noisyGroup.blocks[0].data /= weightSum;
        //
    }
}
Mat bm3d_step2(Mat noisyImage, Mat basicImage) {
    auto ext = noisyImage.size();

    std::vector<BlockGroup> basicBlockGroups;
    for (int j = 0; j < ext.height; j += param_step_2.N_1_wie) {
        for (int i = 0; i < ext.width; i += param_step_2.N_1_wie) {
            auto&& basicGroup = basicBlockGroups.emplace_back();
            step_2_block_matching(basicGroup, basicImage, i, j);
        }
    }
    // construct noisy block group
    int basicGroupCount = basicBlockGroups.size();
    std::vector<BlockGroup> noisyBlockGroups(basicGroupCount);

    for (int i = 0; i < basicGroupCount; ++i) {
        auto&& noisyGroup = noisyBlockGroups[i];
        auto&& basicGroup = basicBlockGroups[i];
        noisyGroup.image = noisyImage;
        noisyGroup.blockWidth = basicGroup.blockWidth;
        noisyGroup.blockHeight = basicGroup.blockHeight;
        int blockCount = basicBlockGroups[i].blocks.size();
        noisyGroup.blocks.resize(blockCount);
        for (int j = 0; j < blockCount; ++j) {
            auto& basicBlock = basicBlockGroups[i].blocks[j];
            auto m = step_2_block(noisyImage, Rect{basicBlock.x, basicBlock.y, param_step_2.N_1_wie,
                                                   param_step_2.N_1_wie});
            noisyGroup.blocks[j] = {0, basicBlock.x, basicBlock.y, m};
        }
    }
    step_2_filter_3d(noisyBlockGroups, basicBlockGroups);

    Mat ret(noisyImage.size(), noisyImage.type());
    bm3d_aggregation(ret, noisyBlockGroups);
    return ret;
}

Mat bm3d(Mat noisyImage) {
    // init parameters
    param_step_1.kaiserWeight = kaiser2DSquare(param_step_1.N_1_ht, param_step_1.beta_ht);
    param_step_2.kaiserWeight = kaiser2DSquare(param_step_2.N_1_wie, param_step_2.beta_wie);

    std::cout << "param_step_1.kaiserWeight: \n" << param_step_1.kaiserWeight << std::endl;
    std::cout << "param_step_2.kaiserWeight: \n" << param_step_2.kaiserWeight << std::endl;

    auto basicImage = bm3d_step1(noisyImage);
    imwrite("basic.png", basicImage);

    auto ret = bm3d_step2(noisyImage, basicImage);
    imwrite("final.png", ret);
    return ret;
}

Mat loadImage(const char* file)
{
    std::string image_path = samples::findFile(file);
    Mat img = imread(image_path);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        assert(0);
        return {};
    }
    return img;
}

int main() {
    // test();
    // return 0;
    createNoisyImage();
    Mat imgOrigin = loadImage("lenna.png");
    Mat imgNoisy = loadImage("output_noisy.png");

    // split channel
    Mat img2[3];
    split(imgNoisy, img2);

    //split first channel
    // convert to float
    Mat noisyImage;
    img2[0].convertTo(noisyImage, CV_32F);

    // split channel
    Mat img3[3];
    split(imgOrigin, img3);

    // convert to float
    Mat originImg;
    img3[0].convertTo(originImg, CV_32F);


    auto finalImage= bm3d(noisyImage);

    auto psnr = PSNR(originImg, finalImage);

    std::cout << "psnr : " << psnr << std::endl;
}