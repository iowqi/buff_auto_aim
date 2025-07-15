#include "hough_circles_alt.hpp"
using namespace cv;
namespace houghtf
{
    void HoughCirclesAlt(const Mat &img, std::vector<EstimatedCircle> &circles, double dp, double rdMinDist,
                         double minRadius, double maxRadius, double cannyThreshold, double minCos2)
    {
        const int MIN_COUNT = 10;
        const int RAY_FP_BITS = 10;
        const int RAY_FP_SCALE = 1 << RAY_FP_BITS;
        const int ACCUM_FP_BITS = 6;
        const int RAY_SHIFT2 = ACCUM_FP_BITS / 2;
        const int ACCUM_ALPHA_ONE = 1 << RAY_SHIFT2;
        const int ACCUM_ALPHA_MASK = ACCUM_ALPHA_ONE - 1;
        const int RAY_SHIFT1 = RAY_FP_BITS - RAY_SHIFT2;
        const int RAY_DELTA1 = 1 << (RAY_SHIFT1 - 1);

        const double ARC_DELTA = 80;
        const double ARC_EPS = 0.03;
        const double CIRCLE_AREA_OFFSET = 4000;
        const double ARC2CLUSTER_EPS = 0.06;
        const double CLUSTER_MERGE_EPS = 0.075;
        const double FINAL_MERGE_DIST_EPS = 0.01;
        const double FINAL_MERGE_AREA_EPS = CLUSTER_MERGE_EPS;

        if (maxRadius <= 0)
            maxRadius = std::min(img.cols, img.rows) * 0.5;
        if (minRadius > maxRadius)
            std::swap(minRadius, maxRadius);
        maxRadius = std::min(maxRadius, std::min(img.cols, img.rows) * 0.5);
        maxRadius = std::max(maxRadius, 1.);
        minRadius = std::max(minRadius, 1.);
        minRadius = std::min(minRadius, maxRadius);
        cannyThreshold = std::max(cannyThreshold, 1.);
        dp = std::max(dp, 1.);

        Mat Dx, Dy, edges;
        // Sobel(img,Dx,CV_16S,1,0);
        // Sobel(img,Dy,CV_16S,0,1);
        Scharr(img, Dx, CV_16S, 1, 0);
        Scharr(img, Dy, CV_16S, 0, 1);
        Canny(Dx, Dy, edges, cannyThreshold / 2, cannyThreshold, true);
        Mat mask(img.rows + 2, img.cols + 2, CV_8U, Scalar::all(0));
        double idp = 1. / dp;
        int minR = cvFloor(minRadius * idp);
        int maxR = cvCeil(maxRadius * idp);
        int acols = cvRound(img.cols * idp);
        int arows = cvRound(img.rows * idp);
        Mat accum(arows + 1, acols + 1, CV_32S, Scalar::all(0));
        int *adata = accum.ptr<int>();
        int astep = (int)accum.step1();
        minR = std::max(minR, 1);
        maxR = std::max(maxR, 1);

        const uchar *edgeData = edges.ptr<uchar>();
        int estep = (int)edges.step1();
        const short *dxData = Dx.ptr<short>();
        const short *dyData = Dy.ptr<short>();
        int dxystep = (int)Dx.step1();
        uchar *mdata = mask.ptr<uchar>();
        int mstep = (int)mask.step1();

        circles.clear();
        std::vector<Vec4f> nz;

        std::vector<Point> stack;
        const int n33[][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}};

        for (int x = 0; x < mask.cols; x++)
            mdata[x] = mdata[(mask.rows - 1) * mstep + x] = (uchar)1;
        for (int y = 0; y < mask.rows; y++)
            mdata[y * mstep] = mdata[y * mstep + mask.cols - 1] = (uchar)1;
        mdata += mstep + 1;

        for (int y = 0; y < edges.rows; y++)
        {
            for (int x = 0; x < edges.cols; x++)
            {
                if (!edgeData[y * estep + x] || mdata[y * mstep + x])
                    continue;

                mdata[y * mstep + x] = (uchar)1;
                stack.push_back(Point(x, y));
                bool backtrace_mode = false;

                do
                {
                    Point p = stack.back();
                    stack.pop_back();
                    int vx = dxData[p.y * dxystep + p.x];
                    int vy = dyData[p.y * dxystep + p.x];

                    float mag = std::sqrt((float)vx * vx + (float)vy * vy);
                    nz.push_back(Vec4f((float)p.x, (float)p.y, (float)vx, (float)vy));
                    CV_Assert(mdata[p.y * mstep + p.x] == 1);

                    int sx = cvRound(vx * RAY_FP_SCALE / mag);
                    int sy = cvRound(vy * RAY_FP_SCALE / mag);

                    int x0 = cvRound((p.x * idp) * RAY_FP_SCALE);
                    int y0 = cvRound((p.y * idp) * RAY_FP_SCALE);

                    // Step from min_radius to max_radius in both directions of the gradient
                    for (int k1 = 0; k1 < 2; k1++)
                    {
                        int x1 = x0 + minR * sx;
                        int y1 = y0 + minR * sy;

                        for (int r = minR; r <= maxR; x1 += sx, y1 += sy, r++)
                        {
                            int x2a = (x1 + RAY_DELTA1) >> RAY_SHIFT1, y2a = (y1 + RAY_DELTA1) >> RAY_SHIFT1;
                            int x2 = x2a >> RAY_SHIFT2, y2 = y2a >> RAY_SHIFT2;
                            if ((unsigned)x2 >= (unsigned)acols ||
                                (unsigned)y2 >= (unsigned)arows)
                                break;

                            // instead of giving everything to the computed pixel of the accumulator,
                            // do a weighted update of 4 neighbor (2x2) pixels using bilinear interpolation.
                            // we do it to reduce the aliasing effect, even though it's slower
                            int *ptr = adata + y2 * astep + x2;
                            int a = (x2a & ACCUM_ALPHA_MASK), b = (y2a & ACCUM_ALPHA_MASK);
                            ptr[0] += (ACCUM_ALPHA_ONE - a) * (ACCUM_ALPHA_ONE - b);
                            ptr[1] += a * (ACCUM_ALPHA_ONE - b);
                            ptr[astep] += (ACCUM_ALPHA_ONE - a) * b;
                            ptr[astep + 1] += a * b;
                        }

                        sx = -sx;
                        sy = -sy;
                    }

                    int neighbors = 0;
                    for (int k = 0; k < 8; k++)
                    {
                        int dy = n33[k][0], dx = n33[k][1];
                        int y_ = p.y + dy, x_ = p.x + dx;
                        if (mdata[y_ * mstep + x_] || !edgeData[y_ * estep + x_])
                            continue;
                        mdata[y_ * mstep + x_] = (uchar)1;
                        stack.push_back(Point(x_, y_));
                        neighbors++;
                    }

                    if (neighbors == 0)
                    {
                        if (backtrace_mode)
                            nz.pop_back();
                        backtrace_mode = true;
                    }
                    else
                        backtrace_mode = false;
                } while (!stack.empty());
                // insert a special "stop marker" in the end of each
                // connected component to make sure we
                // finalize and analyze the arc segment
                nz.push_back(Vec4f(0.f, 0.f, 0.f, 0.f));
            }
        }

        if (nz.empty())
            return;

        // use dilation with massive ((rdMinDisp/dp)*2+1) x ((rdMinDisp/dp)*2+1) kernel.
        // this trick helps us quickly find the local maxima of accumulator value
        // that are at least within the specified distance from each other.
        Mat accum_f, accum_max;
        accum.convertTo(accum_f, CV_32F);
        int niters = std::max(cvCeil(rdMinDist * idp), 1);
        dilate(accum_f, accum_max, Mat(), Point(-1, -1), niters, BORDER_CONSTANT, Scalar::all(0));
        std::vector<Point2f> centers;

        // find the possible circle centers
        for (int y = 0; y < arows; y++)
        {
            const float *adataf = accum_f.ptr<float>(y);
            const float *amaxdata = accum_max.ptr<float>(y);
            int left = -1;
            for (int x = 0; x < acols; x++)
            {
                if (adataf[x] == amaxdata[x] && adataf[x] > adataf[x + astep])
                {
                    if (left < 0)
                        left = x;
                }
                else if (left >= 0)
                {
                    float cx = (float)((left + x - 1) * dp * 0.5f);
                    float cy = (float)(y * dp);
                    centers.push_back(Point2f(cx, cy));
                    left = -1;
                }
            }
        }

        if (centers.empty())
            return;

        float minR2 = (float)(minRadius * minRadius);
        float maxR2 = (float)(maxRadius * maxRadius);
        int nstripes = (int)((centers.size() + HOUGH_CIRCLES_ALT_BLOCK_SIZE - 1) / HOUGH_CIRCLES_ALT_BLOCK_SIZE);
        const int nnz = (int)nz.size();
        Mutex cmutex;

        // Check each possible pair (edge_pixel[i], circle_center[j]).
        // For each circle form the clusters to identify possible radius values.
        // Several clusters (up to 10) are maintained to help to filter out false alarms and
        // to support the concentric circle cases.

        // inside parallel for we process the next "HOUGH_CIRCLES_ALT_BLOCK_SIZE" circles
        parallel_for_(Range(0, nstripes), [&](const Range &r)
                      {
    CircleData cdata[HOUGH_CIRCLES_ALT_BLOCK_SIZE*HOUGH_CIRCLES_ALT_MAX_CLUSTERS];
    CircleData arc[HOUGH_CIRCLES_ALT_BLOCK_SIZE];
    int prev_idx[HOUGH_CIRCLES_ALT_BLOCK_SIZE];

    std::vector<EstimatedCircle> local_circles;
    for(int j0 = r.start*HOUGH_CIRCLES_ALT_BLOCK_SIZE; j0 < r.end*HOUGH_CIRCLES_ALT_BLOCK_SIZE; j0 += HOUGH_CIRCLES_ALT_BLOCK_SIZE)
    {
        const Vec4f* nzdata = &nz[0];
        const Point2f* cc = &centers[j0];
        int nc = std::min((int)(centers.size() - j0), (int)HOUGH_CIRCLES_ALT_BLOCK_SIZE);
        if(nc <= 0) break;

        // reset the statistics about the clusters
        for( int j = 0; j < nc; j++ )
        {
            for( int k = 0; k < HOUGH_CIRCLES_ALT_BLOCK_SIZE; k++ )
                cdata[j*HOUGH_CIRCLES_ALT_MAX_CLUSTERS + k] = CircleData();
            arc[j] = CircleData();
            arc[j].weight = 1; // avoid division by zero
            prev_idx[j] = -2; // we compare the current index "i" with prev_idx[j]+1
                              // to check whether we are still at the current Canny
                              // connected component. so we initially set it to -2
                              // to make sure that the initial check gives "false".
        }

        for( int i = 0; i < nnz; i++ )
        {
            Vec4f v = nzdata[i];
            float x = v[0], y = v[1], vx = v[2], vy = v[3], mag2 = vx*vx + vy*vy;
            bool stop_marker = x == 0.f && y == 0.f && vx == 0.f && vy == 0.f;

            for( int j = 0; j < nc; j++ )
            {
                float cx = cc[j].x, cy = cc[j].y;
                float dx = x - cx, dy = y - cy;
                float rij2 = dx*dx + dy*dy;
                // check that i-th pixel is within the specified distance range from the center
                if( (rij2 > maxR2 || rij2 < minR2) && i < nnz-1 ) continue;
                float dv = dx*vx + dy*vy;
                // check that the line segment connecting the edge pixel and the center and
                // the gradient at the edge pixel are almost collinear
                if( (double)dv*dv < (double)minCos2*mag2*rij2 && i < nnz-1 ) continue;
                float rij = std::sqrt(rij2);

                CircleData& arc_j = arc[j];
                double r_arc = arc_j.rw/arc_j.weight;
                int di0 = 0;
                int prev = prev_idx[j];
                prev_idx[j] = i;

                // update the arc statistics if it still looks like an arc
                if( std::abs(rij - r_arc) < (r_arc + ARC_DELTA)*ARC_EPS && prev+1 == i && !stop_marker )
                {
                    arc_j.rw += rij;
                    arc_j.weight++;
                    di0 = 1;
                    r_arc = arc_j.rw/arc_j.weight;
                    if( i < nnz -1 )
                        continue;
                }

                // otherwise (or in the very end) store the arc in the cluster collection,
                // if the arc is long enough.
                if( arc_j.weight >= MIN_COUNT && arc_j.weight >= r_arc*0.15 )
                {
                    // before doing it, compute the angular range coverage (the mask).
                    uint64 mval = 0;
                    for( int di = 0; di < arc_j.weight; di++ )
                    {
                        int i1 = prev + di0 - di;
                        Vec4f u = nz[i1];
                        float x1 = u[0], y1 = u[1];
                        float dx1 = x1 - cx, dy1 = y1 - cy;
                        float af = fastAtan2(dy1, dx1)*(64.f/360.f);
                        int a = (cvFloor(af) & 63);
                        int b = (a + 1) & 63;
                        af -= a;
                        // this is another protection from aliasing effects
                        if( af <= 0.25f )
                            mval |= (uint64)1 << a;
                        else if( af > 0.75f )
                            mval |= (uint64)1 << b;
                        else
                            mval |= ((uint64)1 << a) | ((uint64)1 << b);
                    }

                    double min_eps = DBL_MAX;
                    int min_mval = (int)(sizeof(mval)*8+1);
                    int k = 0, best_k = -1, subst_k = -1;
                    CircleData* cdata_j = &cdata[j*HOUGH_CIRCLES_ALT_MAX_CLUSTERS];

                    for( ; k < HOUGH_CIRCLES_ALT_MAX_CLUSTERS; k++ )
                    {
                        CircleData& cjk = cdata_j[k];
                        if( cjk.weight == 0 )
                            break;  // it means that there is no more valid clusters
                        double rk = cjk.rw/cjk.weight;
                        // Compute and use the weighted "cluster with arc" area instead of
                        // just cluster area or just arc area or their sum. This is because the cluster can
                        // be small and the arc can be big, or vice versa. Weighted area is more robust.
                        double r2avg = (rk*rk*cjk.weight + r_arc*r_arc*arc_j.weight)/(cjk.weight + arc_j.weight);
                        // It seems to be more robust to compare circle areas (without "pi" scale)
                        // instead of radiuses. When we compare radiuses, when depending on the ALPHA,
                        // different big circles are merged too easily, or different small circles stay different.
                        if( std::abs(rk*rk - r_arc*r_arc) < (r2avg + CIRCLE_AREA_OFFSET)*ARC2CLUSTER_EPS )
                        {
                            double eps = std::abs(rk - r_arc)/rk;
                            if( eps < min_eps )
                            {
                                min_eps = eps;
                                best_k = k;
                            }
                        }
                        else
                        {
                            // Select the cluster with the worst angular coverage.
                            // We use the angular coverage instead of the arc weight
                            // in order to protect real small circles
                            // from "fake" bigger circles with bigger "support".
                            int pcnt = circle_popcnt(cjk.mask);
                            if( pcnt < min_mval )
                            {
                                min_mval = pcnt;
                                subst_k = k;
                            }
                        }
                    }

                    if( best_k >= 0 ) // if found the match, merge the arc into the cluster
                    {
                        CircleData& cjk = cdata_j[best_k];
                        cjk.rw += arc_j.rw;
                        cjk.weight += arc_j.weight;
                        cjk.mask |= mval;
                    }
                    else
                    {
                        if( k < HOUGH_CIRCLES_ALT_MAX_CLUSTERS )
                            subst_k = k; // if we have empty space, just add the new cluster, do not throw anything
                        CircleData& cjk0 = cdata_j[subst_k];

                        // here was the code that attempts to merge the thrown-away cluster with others,
                        // but apparently it does not have any noticeable effect,
                        // so we removed it for the sake of simplicity ...

                        // add the new cluster
                        cjk0.rw = arc_j.rw;
                        cjk0.weight = arc_j.weight;
                        cjk0.mask = mval;
                    }
                }
                // reset the arc statistics.
                arc_j.rw = stop_marker ? 0. : rij;
                arc_j.weight = 1;
                // do not clean arc_j.mval, because we do not alter it.
            }
        }

        // now merge the final clusters for each particular circle center (cx, cy)
        for( int j = 0; j < nc; j++ )
        {
            CircleData* cdata_j = &cdata[j*HOUGH_CIRCLES_ALT_MAX_CLUSTERS];
            float cx = cc[j].x, cy = cc[j].y;

            for( int k = 0; k < HOUGH_CIRCLES_ALT_MAX_CLUSTERS; k++ )
            {
                CircleData& cjk = cdata_j[k];
                if( cjk.weight == 0 )
                    continue;

                // Let in only more or less significant clusters.
                // Small clusters more likely correspond to a noise
                // (otherwise they would grew more substantial during the
                // cluster construction phase).
                // Processing those noisy clusters takes time and
                // potentially decreases accuracy of computed radiuses
                // of good clusters.
                double rjk = cjk.rw/cjk.weight;
                if( cjk.weight < rjk || circle_popcnt(cjk.mask) < 15 )
                    cjk.weight = 0;
            }

            // extensive O(nclusters^2) cluster merge algorithm, but since the number
            // of clusters is limited with a modest constant HOUGH_CIRCLES_ALT_MAX_CLUSTERS,
            // it's still O(1) algorithm :)
            for( int k = 0; k < HOUGH_CIRCLES_ALT_MAX_CLUSTERS; k++ )
            {
                CircleData& cjk = cdata_j[k];
                if( cjk.weight == 0 )
                    continue;
                double rk = cjk.rw/cjk.weight;

                int l = k+1;
                for( ; l < HOUGH_CIRCLES_ALT_MAX_CLUSTERS; l++ )
                {
                    CircleData& cjl = cdata_j[l];
                    if( l == k || cjl.weight == 0 )
                        continue;
                    double rl = cjl.rw/cjl.weight;
                    // Here we use a simple sum of areas (without "pi" scale) instead of weighted
                    // sum just for simplicity and potentially for better accuracy.
                    if( std::abs(rk*rk - rl*rl) < (rk*rk + rl*rl + CIRCLE_AREA_OFFSET)*CLUSTER_MERGE_EPS)
                    {
                        cjk.rw += cjl.rw;
                        cjk.weight += cjl.weight;
                        cjk.mask |= cjl.mask;
                        rk = cjk.rw/cjk.weight;
                        cjl.weight = 0;
                        l = -1; // try to merge other clusters again with the updated k-th cluster
                    }
                }
            }

            for( int k = 0; k < HOUGH_CIRCLES_ALT_MAX_CLUSTERS; k++ )
            {
                CircleData& cjk = cdata_j[k];
                if( cjk.weight == 0 )
                    continue;
                double rk = cjk.rw/cjk.weight;
                uint64 mask_jk = cjk.mask, mask_jk0 = (mask_jk + 1) ^ mask_jk;
                int count = 0, count0 = -1, runlen = 0, max_runlen = 0;
                int prev_bit = 0;
                for( int b = 0; b < 64; b++, mask_jk >>= 1, mask_jk0 >>= 1 )
                {
                    int bit_k = (mask_jk & 1) != 0;
                    count += bit_k;
                    count0 += (mask_jk0 & 1) != 0;
                    if(bit_k == prev_bit) { runlen++; continue; }
                    if(prev_bit == 1)
                        max_runlen = std::max(max_runlen, runlen);
                    runlen = 1;
                    prev_bit = bit_k;
                }
                if( prev_bit == 1)
                    max_runlen = std::max(max_runlen, runlen + (count < 64 ? count0 : 0));

                // Those constants are the results of fine-tuning.
                // Basically, by lowering thresholds more real circles, as well as fake circles, are accepted.
                // By raising the thresholds you get less real circles and less false alarms.
                // A better and more safe way to obtain better detection results is to regulate
                // [minRadius, maxRadius] range and to play with minCos2 parameter.
                // May be some classifier can be trained that takes the weight,
                // circle radius and the bit mask as inputs and produces the verdict.
                bool accepted = (cjk.weight >= rk*3 && count >= 35 && max_runlen >= 20) || count >= 55;
                //if(debug)
                //printf("[%c]. cx=%.1f, cy=%.1f, r=%.1f, weight=%d, count=%d, max_runlen=%d, mask=%016llx\n",
                //       (accepted ? '+' : '-'), cx, cy, rk, cjk.weight, count, max_runlen, cjk.mask);

                if( accepted )
                    local_circles.push_back(EstimatedCircle(Vec3f(cx, cy, (float)rk), cjk.weight));
            }
        }
    }
    if(!local_circles.empty())
    {
        cmutex.lock();
        std::copy(local_circles.begin(), local_circles.end(), std::back_inserter(circles));
        cmutex.unlock();
    } });

        // The final circle merge procedure.
        // This is O(ncircles^2) algorithm
        // and it can take a long time in some specific scenarious.
        // But most of the time it's very fast.
        size_t i0 = 0, nc = circles.size();
        for (size_t i = 0; i < nc; i++)
        {
            if (circles[i].accum == 0)
                continue;
            EstimatedCircle &ci = circles[i0] = circles[i];
            for (size_t j = i + 1; j < nc; j++)
            {
                EstimatedCircle cj = circles[j];
                if (cj.accum == 0)
                    continue;
                float dx = ci.c[0] - cj.c[0], dy = ci.c[1] - cj.c[1];
                float r2 = dx * dx + dy * dy;
                float rs = ci.c[2] + cj.c[2];
                if (r2 > rs * rs * FINAL_MERGE_DIST_EPS)
                    continue;
                if (std::abs(ci.c[2] * ci.c[2] - cj.c[2] * cj.c[2]) <
                    (ci.c[2] * ci.c[2] + cj.c[2] * cj.c[2] + CIRCLE_AREA_OFFSET) * FINAL_MERGE_AREA_EPS)
                {
                    int wi = ci.accum, wj = cj.accum;
                    if (wi < wj)
                        std::swap(ci, cj);
                    circles[j].accum = 0;
                }
            }
            i0++;
        }
        circles.resize(i0);
    }
}