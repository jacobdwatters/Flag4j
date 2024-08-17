package org.flag4j.complex_sparse_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrix;
import org.flag4j.arrays_old.sparse.CooMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.linalg.ops.DirectSum;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooCMatrixDirectSumTests {

    @Test
    void realSparseDirectSumTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        double[] bEntries;
        CooMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new CNumber[]{new CNumber("0.82995+0.51127i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new double[]{0.24295, 0.61213};
        bRowIndices = new int[]{1, 2};
        bColIndices = new int[]{0, 0};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 6);
        expEntries = new CNumber[]{new CNumber("0.82995+0.51127i"), new CNumber("0.24295"), new CNumber("0.61213")};
        expRowIndices = new int[]{0, 3, 4};
        expColIndices = new int[]{1, 3, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new double[]{0.00087, 0.09723};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{1, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 4);
        expEntries = new CNumber[]{new CNumber("0.00087"), new CNumber("0.09723")};
        expRowIndices = new int[]{1, 2};
        expColIndices = new int[]{3, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new CNumber[]{new CNumber("0.70661+0.51154i"), new CNumber("0.327+0.72145i"), new CNumber("0.61639+0.58788i"), new CNumber("0.62248+0.43954i"), new CNumber("0.4152+0.75629i"), new CNumber("0.9972+0.47859i"), new CNumber("0.79032+0.31024i"), new CNumber("0.67413+0.62249i"), new CNumber("0.79157+0.41042i"), new CNumber("0.85885+0.02278i"), new CNumber("0.18961+0.82326i"), new CNumber("0.22331+0.54411i"), new CNumber("0.3942+0.44593i"), new CNumber("0.00387+0.70256i")};
        aRowIndices = new int[]{0, 0, 1, 3, 5, 6, 7, 9, 11, 11, 11, 12, 13, 13};
        aColIndices = new int[]{2, 3, 4, 0, 0, 2, 1, 1, 0, 3, 4, 0, 1, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new double[]{0.28292, 0.78377, 0.55366, 0.29866, 0.27736, 0.11891, 0.75654, 0.50066, 0.81601, 0.41473, 0.82755, 0.36692, 0.51807, 0.96648, 0.97952, 0.25537, 0.05777};
        bRowIndices = new int[]{0, 0, 1, 2, 4, 4, 5, 5, 5, 8, 9, 10, 11, 12, 12, 13, 13};
        bColIndices = new int[]{1, 2, 1, 5, 0, 1, 1, 4, 5, 3, 3, 2, 3, 2, 4, 0, 3};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(28, 11);
        expEntries = new CNumber[]{new CNumber("0.70661+0.51154i"), new CNumber("0.327+0.72145i"), new CNumber("0.61639+0.58788i"), new CNumber("0.62248+0.43954i"), new CNumber("0.4152+0.75629i"), new CNumber("0.9972+0.47859i"), new CNumber("0.79032+0.31024i"), new CNumber("0.67413+0.62249i"), new CNumber("0.79157+0.41042i"), new CNumber("0.85885+0.02278i"), new CNumber("0.18961+0.82326i"), new CNumber("0.22331+0.54411i"), new CNumber("0.3942+0.44593i"), new CNumber("0.00387+0.70256i"), new CNumber("0.28292"), new CNumber("0.78377"), new CNumber("0.55366"), new CNumber("0.29866"), new CNumber("0.27736"), new CNumber("0.11891"), new CNumber("0.75654"), new CNumber("0.50066"), new CNumber("0.81601"), new CNumber("0.41473"), new CNumber("0.82755"), new CNumber("0.36692"), new CNumber("0.51807"), new CNumber("0.96648"), new CNumber("0.97952"), new CNumber("0.25537"), new CNumber("0.05777")};
        expRowIndices = new int[]{0, 0, 1, 3, 5, 6, 7, 9, 11, 11, 11, 12, 13, 13, 14, 14, 15, 16, 18, 18, 19, 19, 19, 22, 23, 24, 25, 26, 26, 27, 27};
        expColIndices = new int[]{2, 3, 4, 0, 0, 2, 1, 1, 0, 3, 4, 0, 1, 3, 6, 7, 6, 10, 5, 6, 6, 9, 10, 8, 8, 7, 8, 7, 9, 5, 8};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));
    }


    @Test
    void complexSparseDirectSumTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        CNumber[] bEntries;
        CooCMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new CNumber[]{new CNumber("0.08806+0.0553i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new CNumber[]{new CNumber("0.31387+0.81903i"), new CNumber("0.66089+0.14127i")};
        bRowIndices = new int[]{0, 0};
        bColIndices = new int[]{0, 1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 6);
        expEntries = new CNumber[]{new CNumber("0.08806+0.0553i"), new CNumber("0.31387+0.81903i"), new CNumber("0.66089+0.14127i")};
        expRowIndices = new int[]{0, 2, 2};
        expColIndices = new int[]{1, 3, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new CNumber[]{new CNumber("0.6257+0.92994i"), new CNumber("0.79307+0.00425i")};
        bRowIndices = new int[]{1, 3};
        bColIndices = new int[]{0, 1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 4);
        expEntries = new CNumber[]{new CNumber("0.6257+0.92994i"), new CNumber("0.79307+0.00425i")};
        expRowIndices = new int[]{2, 4};
        expColIndices = new int[]{2, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new CNumber[]{new CNumber("0.05752+0.87101i"), new CNumber("0.5312+0.44755i"), new CNumber("0.67819+0.50842i"), new CNumber("0.65475+0.47186i"), new CNumber("0.36484+0.79259i"), new CNumber("0.74912+0.24418i"), new CNumber("0.89577+0.84938i"), new CNumber("0.24992+0.05502i"), new CNumber("0.85454+0.04111i"), new CNumber("0.06607+0.70307i"), new CNumber("0.66633+0.53082i"), new CNumber("0.04777+0.6303i"), new CNumber("0.74899+0.22239i"), new CNumber("0.99873+0.13431i")};
        aRowIndices = new int[]{1, 2, 3, 3, 5, 6, 6, 6, 9, 11, 11, 12, 13, 13};
        aColIndices = new int[]{0, 1, 0, 1, 0, 1, 2, 4, 4, 1, 3, 3, 1, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new CNumber[]{new CNumber("0.41166+0.63871i"), new CNumber("0.39402+0.53672i"), new CNumber("0.43344+0.54967i"), new CNumber("0.22261+0.31397i"), new CNumber("0.5524+0.78872i"), new CNumber("0.22091+0.08416i"), new CNumber("0.88303+0.06704i"), new CNumber("0.55587+0.31015i"), new CNumber("0.42591+0.83689i"), new CNumber("0.8109+0.26634i"), new CNumber("0.39993+0.2489i"), new CNumber("0.91958+0.95841i"), new CNumber("0.98445+0.78723i"), new CNumber("0.29625+0.77089i"), new CNumber("0.56789+0.45746i"), new CNumber("0.30379+0.88106i"), new CNumber("0.00235+0.57188i")};
        bRowIndices = new int[]{0, 1, 1, 2, 2, 3, 3, 3, 6, 7, 7, 7, 10, 11, 11, 12, 12};
        bColIndices = new int[]{0, 2, 4, 3, 4, 0, 4, 5, 5, 2, 3, 4, 4, 0, 4, 3, 5};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(28, 11);
        expEntries = new CNumber[]{new CNumber("0.05752+0.87101i"), new CNumber("0.5312+0.44755i"), new CNumber("0.67819+0.50842i"), new CNumber("0.65475+0.47186i"), new CNumber("0.36484+0.79259i"), new CNumber("0.74912+0.24418i"), new CNumber("0.89577+0.84938i"), new CNumber("0.24992+0.05502i"), new CNumber("0.85454+0.04111i"), new CNumber("0.06607+0.70307i"), new CNumber("0.66633+0.53082i"), new CNumber("0.04777+0.6303i"), new CNumber("0.74899+0.22239i"), new CNumber("0.99873+0.13431i"), new CNumber("0.41166+0.63871i"), new CNumber("0.39402+0.53672i"), new CNumber("0.43344+0.54967i"), new CNumber("0.22261+0.31397i"), new CNumber("0.5524+0.78872i"), new CNumber("0.22091+0.08416i"), new CNumber("0.88303+0.06704i"), new CNumber("0.55587+0.31015i"), new CNumber("0.42591+0.83689i"), new CNumber("0.8109+0.26634i"), new CNumber("0.39993+0.2489i"), new CNumber("0.91958+0.95841i"), new CNumber("0.98445+0.78723i"), new CNumber("0.29625+0.77089i"), new CNumber("0.56789+0.45746i"), new CNumber("0.30379+0.88106i"), new CNumber("0.00235+0.57188i")};
        expRowIndices = new int[]{1, 2, 3, 3, 5, 6, 6, 6, 9, 11, 11, 12, 13, 13, 14, 15, 15, 16, 16, 17, 17, 17, 20, 21, 21, 21, 24, 25, 25, 26, 26};
        expColIndices = new int[]{0, 1, 0, 1, 0, 1, 2, 4, 4, 1, 3, 3, 1, 3, 5, 7, 9, 8, 9, 5, 9, 10, 10, 7, 8, 9, 9, 5, 9, 8, 10};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));
    }


    @Test
    void realDenseDirectSumTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        double[][] bEntries;
        MatrixOld b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new CNumber[]{new CNumber("0.75308+0.91095i")};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.09549, 0.7115, 0.91745},
                {0.90874, 0.61103, 0.03209},
                {0.23889, 0.53504, 0.6039},
                {0.52828, 0.16486, 0.74791}};
        b = new MatrixOld(bEntries);

        expShape = new Shape(6, 6);
        expEntries = new CNumber[]{new CNumber("0.75308+0.91095i"), new CNumber("0.09549"), new CNumber("0.7115"), new CNumber("0.91745"), new CNumber("0.90874"), new CNumber("0.61103"), new CNumber("0.03209"), new CNumber("0.23889"), new CNumber("0.53504"), new CNumber("0.6039"), new CNumber("0.52828"), new CNumber("0.16486"), new CNumber("0.74791")};
        expRowIndices = new int[]{1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5};
        expColIndices = new int[]{2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.19876, 0.89365},
                {0.55613, 0.74748},
                {0.96712, 0.1781},
                {0.66665, 0.9961},
                {0.81941, 0.34338}};
        b = new MatrixOld(bEntries);

        expShape = new Shape(6, 4);
        expEntries = new CNumber[]{new CNumber("0.19876"), new CNumber("0.89365"), new CNumber("0.55613"), new CNumber("0.74748"), new CNumber("0.96712"), new CNumber("0.1781"), new CNumber("0.66665"), new CNumber("0.9961"), new CNumber("0.81941"), new CNumber("0.34338")};
        expRowIndices = new int[]{1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
        expColIndices = new int[]{2, 3, 2, 3, 2, 3, 2, 3, 2, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new CNumber[]{new CNumber("0.30839+0.40829i"), new CNumber("0.39726+0.35016i"), new CNumber("0.8527+0.62435i"), new CNumber("0.98366+0.35628i"), new CNumber("0.88925+0.67872i"), new CNumber("0.32378+0.81966i"), new CNumber("0.3507+0.84697i"), new CNumber("0.8837+0.93694i"), new CNumber("0.26869+0.33957i"), new CNumber("0.27441+0.37573i"), new CNumber("0.61453+0.26505i"), new CNumber("0.77154+0.88759i"), new CNumber("0.18096+0.31388i"), new CNumber("0.07751+0.22487i")};
        aRowIndices = new int[]{0, 0, 1, 4, 5, 7, 8, 8, 8, 9, 9, 10, 10, 13};
        aColIndices = new int[]{1, 4, 4, 2, 1, 0, 0, 1, 3, 0, 1, 3, 4, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.79875, 0.60342, 0.64119, 0.99318, 0.62621, 0.96078},
                {0.57069, 0.9969, 0.59859, 0.57978, 0.74878, 0.19627},
                {0.54943, 0.28993, 0.04304, 0.71078, 0.17952, 0.95647},
                {0.06866, 0.40802, 0.10196, 0.228, 0.64171, 0.20432},
                {0.18203, 0.84139, 0.57641, 0.46874, 0.79593, 0.34331},
                {0.84094, 0.98363, 0.76836, 0.24771, 0.07311, 0.60237},
                {0.16047, 0.90748, 0.25158, 0.59591, 0.7683, 0.51682},
                {0.61042, 0.8872, 0.75956, 0.44605, 0.021, 0.65854},
                {0.43082, 0.15762, 0.57832, 0.08147, 0.96463, 0.90699},
                {0.60277, 0.94198, 0.67233, 0.73323, 0.80128, 0.67584},
                {0.53108, 0.95543, 0.54357, 0.12398, 0.37815, 0.92178},
                {0.99005, 0.9346, 0.69267, 0.60716, 0.87631, 0.26384},
                {0.3144, 0.07245, 0.62588, 0.75088, 0.36216, 0.54433},
                {0.79115, 0.92417, 0.92747, 0.4773, 0.72474, 0.43361}};
        b = new MatrixOld(bEntries);

        expShape = new Shape(28, 11);
        expEntries = new CNumber[]{new CNumber("0.30839+0.40829i"), new CNumber("0.39726+0.35016i"), new CNumber("0.8527+0.62435i"), new CNumber("0.98366+0.35628i"), new CNumber("0.88925+0.67872i"), new CNumber("0.32378+0.81966i"), new CNumber("0.3507+0.84697i"), new CNumber("0.8837+0.93694i"), new CNumber("0.26869+0.33957i"), new CNumber("0.27441+0.37573i"), new CNumber("0.61453+0.26505i"), new CNumber("0.77154+0.88759i"), new CNumber("0.18096+0.31388i"), new CNumber("0.07751+0.22487i"), new CNumber("0.79875"), new CNumber("0.60342"), new CNumber("0.64119"), new CNumber("0.99318"), new CNumber("0.62621"), new CNumber("0.96078"), new CNumber("0.57069"), new CNumber("0.9969"), new CNumber("0.59859"), new CNumber("0.57978"), new CNumber("0.74878"), new CNumber("0.19627"), new CNumber("0.54943"), new CNumber("0.28993"), new CNumber("0.04304"), new CNumber("0.71078"), new CNumber("0.17952"), new CNumber("0.95647"), new CNumber("0.06866"), new CNumber("0.40802"), new CNumber("0.10196"), new CNumber("0.228"), new CNumber("0.64171"), new CNumber("0.20432"), new CNumber("0.18203"), new CNumber("0.84139"), new CNumber("0.57641"), new CNumber("0.46874"), new CNumber("0.79593"), new CNumber("0.34331"), new CNumber("0.84094"), new CNumber("0.98363"), new CNumber("0.76836"), new CNumber("0.24771"), new CNumber("0.07311"), new CNumber("0.60237"), new CNumber("0.16047"), new CNumber("0.90748"), new CNumber("0.25158"), new CNumber("0.59591"), new CNumber("0.7683"), new CNumber("0.51682"), new CNumber("0.61042"), new CNumber("0.8872"), new CNumber("0.75956"), new CNumber("0.44605"), new CNumber("0.021"), new CNumber("0.65854"), new CNumber("0.43082"), new CNumber("0.15762"), new CNumber("0.57832"), new CNumber("0.08147"), new CNumber("0.96463"), new CNumber("0.90699"), new CNumber("0.60277"), new CNumber("0.94198"), new CNumber("0.67233"), new CNumber("0.73323"), new CNumber("0.80128"), new CNumber("0.67584"), new CNumber("0.53108"), new CNumber("0.95543"), new CNumber("0.54357"), new CNumber("0.12398"), new CNumber("0.37815"), new CNumber("0.92178"), new CNumber("0.99005"), new CNumber("0.9346"), new CNumber("0.69267"), new CNumber("0.60716"), new CNumber("0.87631"), new CNumber("0.26384"), new CNumber("0.3144"), new CNumber("0.07245"), new CNumber("0.62588"), new CNumber("0.75088"), new CNumber("0.36216"), new CNumber("0.54433"), new CNumber("0.79115"), new CNumber("0.92417"), new CNumber("0.92747"), new CNumber("0.4773"), new CNumber("0.72474"), new CNumber("0.43361")};
        expRowIndices = new int[]{0, 0, 1, 4, 5, 7, 8, 8, 8, 9, 9, 10, 10, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27};
        expColIndices = new int[]{1, 4, 4, 2, 1, 0, 0, 1, 3, 0, 1, 3, 4, 0, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));
    }


    @Test
    void complexDenseDirectSumTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        CNumber[][] bEntries;
        CMatrixOld b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new CNumber[]{new CNumber("0.44357+0.56089i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.78452+0.56837i"), new CNumber("0.6448+0.79154i"), new CNumber("0.06747+0.21863i")},
                {new CNumber("0.02255+0.7866i"), new CNumber("0.98096+0.80129i"), new CNumber("0.19257+0.07889i")},
                {new CNumber("0.12196+0.8614i"), new CNumber("0.81582+0.64689i"), new CNumber("0.38959+0.17485i")},
                {new CNumber("0.94487+0.98354i"), new CNumber("0.34796+0.47073i"), new CNumber("0.78694+0.08697i")}};
        b = new CMatrixOld(bEntries);

        expShape = new Shape(6, 6);
        expEntries = new CNumber[]{new CNumber("0.44357+0.56089i"), new CNumber("0.78452+0.56837i"), new CNumber("0.6448+0.79154i"), new CNumber("0.06747+0.21863i"), new CNumber("0.02255+0.7866i"), new CNumber("0.98096+0.80129i"), new CNumber("0.19257+0.07889i"), new CNumber("0.12196+0.8614i"), new CNumber("0.81582+0.64689i"), new CNumber("0.38959+0.17485i"), new CNumber("0.94487+0.98354i"), new CNumber("0.34796+0.47073i"), new CNumber("0.78694+0.08697i")};
        expRowIndices = new int[]{0, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5};
        expColIndices = new int[]{2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.15294+0.06902i"), new CNumber("0.03597+0.82551i")},
                {new CNumber("0.27583+0.49029i"), new CNumber("0.28027+0.53915i")},
                {new CNumber("0.4234+0.47288i"), new CNumber("0.55953+0.60797i")},
                {new CNumber("0.31641+0.00323i"), new CNumber("0.32307+0.84634i")},
                {new CNumber("0.44636+0.70976i"), new CNumber("0.57115+0.24951i")}};
        b = new CMatrixOld(bEntries);

        expShape = new Shape(6, 4);
        expEntries = new CNumber[]{new CNumber("0.15294+0.06902i"), new CNumber("0.03597+0.82551i"), new CNumber("0.27583+0.49029i"), new CNumber("0.28027+0.53915i"), new CNumber("0.4234+0.47288i"), new CNumber("0.55953+0.60797i"), new CNumber("0.31641+0.00323i"), new CNumber("0.32307+0.84634i"), new CNumber("0.44636+0.70976i"), new CNumber("0.57115+0.24951i")};
        expRowIndices = new int[]{1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
        expColIndices = new int[]{2, 3, 2, 3, 2, 3, 2, 3, 2, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new CNumber[]{new CNumber("0.40337+0.18358i"), new CNumber("0.23201+0.12493i"), new CNumber("0.80599+0.36797i"), new CNumber("0.6291+0.04377i"), new CNumber("0.61211+0.512i"), new CNumber("0.30291+0.94184i"), new CNumber("0.5474+0.10516i"), new CNumber("0.66737+0.36392i"), new CNumber("0.4509+0.38233i"), new CNumber("0.79195+0.53952i"), new CNumber("0.2584+0.99285i"), new CNumber("0.45507+0.29298i"), new CNumber("0.95008+0.59176i"), new CNumber("0.907+0.21149i")};
        aRowIndices = new int[]{0, 2, 3, 5, 6, 6, 7, 7, 8, 9, 9, 9, 11, 12};
        aColIndices = new int[]{0, 0, 0, 1, 0, 1, 1, 2, 1, 0, 2, 3, 1, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.60006+0.24167i"), new CNumber("0.659+0.77568i"), new CNumber("0.78701+0.97131i"), new CNumber("0.1826+0.32818i"), new CNumber("0.63579+0.2251i"), new CNumber("0.53748+0.35045i")},
                {new CNumber("0.98483+0.56784i"), new CNumber("0.5851+0.8565i"), new CNumber("0.22168+0.98113i"), new CNumber("0.07129+0.60116i"), new CNumber("0.18652+0.37409i"), new CNumber("0.36718+0.4149i")},
                {new CNumber("0.40068+0.3205i"), new CNumber("0.2379+0.87231i"), new CNumber("0.16795+0.80401i"), new CNumber("0.26758+0.54973i"), new CNumber("0.28903+0.67547i"), new CNumber("0.96197+0.60531i")},
                {new CNumber("0.91218+0.54128i"), new CNumber("0.53681+0.37373i"), new CNumber("0.32929+0.8857i"), new CNumber("0.62606+0.35994i"), new CNumber("0.70395+0.77007i"), new CNumber("0.8333+0.42747i")},
                {new CNumber("0.53045+0.2778i"), new CNumber("0.20728+0.41615i"), new CNumber("0.10019+0.01404i"), new CNumber("0.127+0.54654i"), new CNumber("0.78518+0.11724i"), new CNumber("0.91299+0.27997i")},
                {new CNumber("0.75646+0.49874i"), new CNumber("0.64499+0.9044i"), new CNumber("0.01597+0.68196i"), new CNumber("0.61345+0.6926i"), new CNumber("0.04948+0.3347i"), new CNumber("0.95992+0.05412i")},
                {new CNumber("0.419+0.19859i"), new CNumber("0.63077+0.06182i"), new CNumber("0.92818+0.35942i"), new CNumber("0.61946+0.20906i"), new CNumber("0.80354+0.93532i"), new CNumber("0.51257+0.95759i")},
                {new CNumber("0.12664+0.29012i"), new CNumber("0.19695+0.82645i"), new CNumber("0.65961+0.77686i"), new CNumber("0.69553+0.18088i"), new CNumber("0.3591+0.92494i"), new CNumber("0.35493+0.40976i")},
                {new CNumber("0.86843+0.37022i"), new CNumber("0.99417+0.01138i"), new CNumber("0.00174+0.33098i"), new CNumber("0.3424+0.5745i"), new CNumber("0.97224+0.82872i"), new CNumber("0.61287+0.19762i")},
                {new CNumber("0.1762+0.61634i"), new CNumber("0.94205+0.36257i"), new CNumber("0.0999+0.63339i"), new CNumber("0.37805+0.21013i"), new CNumber("0.62448+0.08611i"), new CNumber("0.50461+0.21309i")},
                {new CNumber("0.68105+0.01474i"), new CNumber("0.52112+0.10812i"), new CNumber("0.88682+0.30062i"), new CNumber("0.42731+0.14287i"), new CNumber("0.55232+0.7957i"), new CNumber("0.31387+0.70592i")},
                {new CNumber("0.17723+0.44997i"), new CNumber("0.95644+0.4034i"), new CNumber("0.47098+0.0609i"), new CNumber("0.06995+0.87607i"), new CNumber("0.09073+0.72433i"), new CNumber("0.03014+0.22533i")},
                {new CNumber("0.7567+0.5915i"), new CNumber("0.85008+0.08665i"), new CNumber("0.70947+0.73309i"), new CNumber("0.57318+0.43883i"), new CNumber("0.63709+0.72072i"), new CNumber("0.14587+0.28509i")},
                {new CNumber("0.63656+0.96017i"), new CNumber("0.96489+0.14421i"), new CNumber("0.79621+0.10729i"), new CNumber("0.46614+0.99075i"), new CNumber("0.68914+0.86662i"), new CNumber("0.92109+0.18268i")}};
        b = new CMatrixOld(bEntries);

        expShape = new Shape(28, 11);
        expEntries = new CNumber[]{new CNumber("0.40337+0.18358i"), new CNumber("0.23201+0.12493i"), new CNumber("0.80599+0.36797i"), new CNumber("0.6291+0.04377i"), new CNumber("0.61211+0.512i"), new CNumber("0.30291+0.94184i"), new CNumber("0.5474+0.10516i"), new CNumber("0.66737+0.36392i"), new CNumber("0.4509+0.38233i"), new CNumber("0.79195+0.53952i"), new CNumber("0.2584+0.99285i"), new CNumber("0.45507+0.29298i"), new CNumber("0.95008+0.59176i"), new CNumber("0.907+0.21149i"), new CNumber("0.60006+0.24167i"), new CNumber("0.659+0.77568i"), new CNumber("0.78701+0.97131i"), new CNumber("0.1826+0.32818i"), new CNumber("0.63579+0.2251i"), new CNumber("0.53748+0.35045i"), new CNumber("0.98483+0.56784i"), new CNumber("0.5851+0.8565i"), new CNumber("0.22168+0.98113i"), new CNumber("0.07129+0.60116i"), new CNumber("0.18652+0.37409i"), new CNumber("0.36718+0.4149i"), new CNumber("0.40068+0.3205i"), new CNumber("0.2379+0.87231i"), new CNumber("0.16795+0.80401i"), new CNumber("0.26758+0.54973i"), new CNumber("0.28903+0.67547i"), new CNumber("0.96197+0.60531i"), new CNumber("0.91218+0.54128i"), new CNumber("0.53681+0.37373i"), new CNumber("0.32929+0.8857i"), new CNumber("0.62606+0.35994i"), new CNumber("0.70395+0.77007i"), new CNumber("0.8333+0.42747i"), new CNumber("0.53045+0.2778i"), new CNumber("0.20728+0.41615i"), new CNumber("0.10019+0.01404i"), new CNumber("0.127+0.54654i"), new CNumber("0.78518+0.11724i"), new CNumber("0.91299+0.27997i"), new CNumber("0.75646+0.49874i"), new CNumber("0.64499+0.9044i"), new CNumber("0.01597+0.68196i"), new CNumber("0.61345+0.6926i"), new CNumber("0.04948+0.3347i"), new CNumber("0.95992+0.05412i"), new CNumber("0.419+0.19859i"), new CNumber("0.63077+0.06182i"), new CNumber("0.92818+0.35942i"), new CNumber("0.61946+0.20906i"), new CNumber("0.80354+0.93532i"), new CNumber("0.51257+0.95759i"), new CNumber("0.12664+0.29012i"), new CNumber("0.19695+0.82645i"), new CNumber("0.65961+0.77686i"), new CNumber("0.69553+0.18088i"), new CNumber("0.3591+0.92494i"), new CNumber("0.35493+0.40976i"), new CNumber("0.86843+0.37022i"), new CNumber("0.99417+0.01138i"), new CNumber("0.00174+0.33098i"), new CNumber("0.3424+0.5745i"), new CNumber("0.97224+0.82872i"), new CNumber("0.61287+0.19762i"), new CNumber("0.1762+0.61634i"), new CNumber("0.94205+0.36257i"), new CNumber("0.0999+0.63339i"), new CNumber("0.37805+0.21013i"), new CNumber("0.62448+0.08611i"), new CNumber("0.50461+0.21309i"), new CNumber("0.68105+0.01474i"), new CNumber("0.52112+0.10812i"), new CNumber("0.88682+0.30062i"), new CNumber("0.42731+0.14287i"), new CNumber("0.55232+0.7957i"), new CNumber("0.31387+0.70592i"), new CNumber("0.17723+0.44997i"), new CNumber("0.95644+0.4034i"), new CNumber("0.47098+0.0609i"), new CNumber("0.06995+0.87607i"), new CNumber("0.09073+0.72433i"), new CNumber("0.03014+0.22533i"), new CNumber("0.7567+0.5915i"), new CNumber("0.85008+0.08665i"), new CNumber("0.70947+0.73309i"), new CNumber("0.57318+0.43883i"), new CNumber("0.63709+0.72072i"), new CNumber("0.14587+0.28509i"), new CNumber("0.63656+0.96017i"), new CNumber("0.96489+0.14421i"), new CNumber("0.79621+0.10729i"), new CNumber("0.46614+0.99075i"), new CNumber("0.68914+0.86662i"), new CNumber("0.92109+0.18268i")};
        expRowIndices = new int[]{0, 2, 3, 5, 6, 6, 7, 7, 8, 9, 9, 9, 11, 12, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27};
        expColIndices = new int[]{0, 0, 0, 1, 0, 1, 1, 2, 1, 0, 2, 3, 1, 0, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.directSum(a, b));
    }


    @Test
    void realSparseInvDirectSumTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        double[] bEntries;
        CooMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new CNumber[]{new CNumber("0.1553+0.01646i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new double[]{0.83496, 0.0695};
        bRowIndices = new int[]{0, 0};
        bColIndices = new int[]{0, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 6);
        expEntries = new CNumber[]{new CNumber("0.83496"), new CNumber("0.0695"), new CNumber("0.1553+0.01646i")};
        expRowIndices = new int[]{0, 0, 4};
        expColIndices = new int[]{3, 4, 2};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new double[]{0.14725, 0.72385};
        bRowIndices = new int[]{2, 3};
        bColIndices = new int[]{0, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 4);
        expEntries = new CNumber[]{new CNumber("0.14725"), new CNumber("0.72385")};
        expRowIndices = new int[]{2, 3};
        expColIndices = new int[]{2, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new CNumber[]{new CNumber("0.98694+0.59681i"), new CNumber("0.22165+0.57898i"), new CNumber("0.25137+0.81555i"), new CNumber("0.01859+0.32605i"), new CNumber("0.52322+0.82209i"), new CNumber("0.4839+0.20329i"), new CNumber("0.12876+0.35791i"), new CNumber("0.42228+0.02274i"), new CNumber("0.60988+0.12416i"), new CNumber("0.88629+0.59152i"), new CNumber("0.74891+0.04819i"), new CNumber("0.85096+0.22109i"), new CNumber("0.79085+0.84112i"), new CNumber("0.84963+0.87431i")};
        aRowIndices = new int[]{4, 4, 5, 6, 6, 6, 7, 8, 9, 11, 11, 12, 12, 13};
        aColIndices = new int[]{0, 2, 1, 2, 3, 4, 1, 0, 4, 2, 3, 1, 2, 4};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new double[]{0.17275, 0.17942, 0.60881, 0.0527, 0.83143, 0.86957, 0.04506, 0.54834, 0.87852, 0.25451, 0.33953, 0.13206, 0.08406, 0.64662, 0.26429, 0.69028, 0.49608};
        bRowIndices = new int[]{0, 1, 1, 2, 4, 6, 7, 7, 8, 8, 9, 11, 11, 12, 12, 12, 13};
        bColIndices = new int[]{0, 0, 3, 1, 3, 0, 0, 3, 0, 1, 5, 0, 3, 1, 2, 4, 2};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(28, 11);
        expEntries = new CNumber[]{new CNumber("0.17275"), new CNumber("0.17942"), new CNumber("0.60881"), new CNumber("0.0527"), new CNumber("0.83143"), new CNumber("0.86957"), new CNumber("0.04506"), new CNumber("0.54834"), new CNumber("0.87852"), new CNumber("0.25451"), new CNumber("0.33953"), new CNumber("0.13206"), new CNumber("0.08406"), new CNumber("0.64662"), new CNumber("0.26429"), new CNumber("0.69028"), new CNumber("0.49608"), new CNumber("0.98694+0.59681i"), new CNumber("0.22165+0.57898i"), new CNumber("0.25137+0.81555i"), new CNumber("0.01859+0.32605i"), new CNumber("0.52322+0.82209i"), new CNumber("0.4839+0.20329i"), new CNumber("0.12876+0.35791i"), new CNumber("0.42228+0.02274i"), new CNumber("0.60988+0.12416i"), new CNumber("0.88629+0.59152i"), new CNumber("0.74891+0.04819i"), new CNumber("0.85096+0.22109i"), new CNumber("0.79085+0.84112i"), new CNumber("0.84963+0.87431i")};
        expRowIndices = new int[]{0, 1, 1, 2, 4, 6, 7, 7, 8, 8, 9, 11, 11, 12, 12, 12, 13, 18, 18, 19, 20, 20, 20, 21, 22, 23, 25, 25, 26, 26, 27};
        expColIndices = new int[]{5, 5, 8, 6, 8, 5, 5, 8, 5, 6, 10, 5, 8, 6, 7, 9, 7, 0, 2, 1, 2, 3, 4, 1, 0, 4, 2, 3, 1, 2, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));
    }


    @Test
    void complexSparseInvDirectSumTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        CNumber[] bEntries;
        CooCMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new CNumber[]{new CNumber("0.98992+0.87766i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new CNumber[]{new CNumber("0.21379+0.47105i"), new CNumber("0.13205+0.07179i")};
        bRowIndices = new int[]{0, 3};
        bColIndices = new int[]{2, 0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 6);
        expEntries = new CNumber[]{new CNumber("0.21379+0.47105i"), new CNumber("0.13205+0.07179i"), new CNumber("0.98992+0.87766i")};
        expRowIndices = new int[]{0, 3, 4};
        expColIndices = new int[]{5, 3, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new CNumber[]{new CNumber("0.48115+0.64388i"), new CNumber("0.41238+0.86793i")};
        bRowIndices = new int[]{1, 3};
        bColIndices = new int[]{0, 0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 4);
        expEntries = new CNumber[]{new CNumber("0.48115+0.64388i"), new CNumber("0.41238+0.86793i")};
        expRowIndices = new int[]{1, 3};
        expColIndices = new int[]{2, 2};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new CNumber[]{new CNumber("0.76075+0.0571i"), new CNumber("0.33837+0.76502i"), new CNumber("0.94177+0.83892i"), new CNumber("0.20583+0.94384i"), new CNumber("0.91332+0.56232i"), new CNumber("0.51064+0.51166i"), new CNumber("0.94406+0.05894i"), new CNumber("0.82868+0.13105i"), new CNumber("0.78945+0.98172i"), new CNumber("0.8299+0.80277i"), new CNumber("0.66449+0.09316i"), new CNumber("0.20635+0.05798i"), new CNumber("0.53974+0.68971i"), new CNumber("0.63979+0.21648i")};
        aRowIndices = new int[]{0, 0, 1, 2, 3, 4, 5, 9, 9, 10, 10, 13, 13, 13};
        aColIndices = new int[]{1, 4, 4, 1, 0, 1, 2, 3, 4, 2, 4, 1, 3, 4};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new CNumber[]{new CNumber("0.76249+0.48371i"), new CNumber("0.69754+0.45912i"), new CNumber("0.80273+0.18843i"), new CNumber("0.82503+0.34889i"), new CNumber("0.16633+0.27299i"), new CNumber("0.22618+0.75381i"), new CNumber("0.79741+0.27608i"), new CNumber("0.35974+0.93911i"), new CNumber("0.90338+0.71516i"), new CNumber("0.60645+0.09484i"), new CNumber("0.40915+0.48148i"), new CNumber("0.80957+0.95002i"), new CNumber("0.77837+0.62861i"), new CNumber("0.41939+0.81049i"), new CNumber("0.63162+0.2394i"), new CNumber("0.44776+0.99792i"), new CNumber("0.17577+0.41738i")};
        bRowIndices = new int[]{0, 0, 2, 2, 3, 4, 4, 5, 5, 5, 9, 10, 11, 11, 12, 12, 13};
        bColIndices = new int[]{0, 3, 1, 4, 3, 2, 5, 2, 4, 5, 5, 3, 1, 4, 2, 3, 5};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(28, 11);
        expEntries = new CNumber[]{new CNumber("0.76249+0.48371i"), new CNumber("0.69754+0.45912i"), new CNumber("0.80273+0.18843i"), new CNumber("0.82503+0.34889i"), new CNumber("0.16633+0.27299i"), new CNumber("0.22618+0.75381i"), new CNumber("0.79741+0.27608i"), new CNumber("0.35974+0.93911i"), new CNumber("0.90338+0.71516i"), new CNumber("0.60645+0.09484i"), new CNumber("0.40915+0.48148i"), new CNumber("0.80957+0.95002i"), new CNumber("0.77837+0.62861i"), new CNumber("0.41939+0.81049i"), new CNumber("0.63162+0.2394i"), new CNumber("0.44776+0.99792i"), new CNumber("0.17577+0.41738i"), new CNumber("0.76075+0.0571i"), new CNumber("0.33837+0.76502i"), new CNumber("0.94177+0.83892i"), new CNumber("0.20583+0.94384i"), new CNumber("0.91332+0.56232i"), new CNumber("0.51064+0.51166i"), new CNumber("0.94406+0.05894i"), new CNumber("0.82868+0.13105i"), new CNumber("0.78945+0.98172i"), new CNumber("0.8299+0.80277i"), new CNumber("0.66449+0.09316i"), new CNumber("0.20635+0.05798i"), new CNumber("0.53974+0.68971i"), new CNumber("0.63979+0.21648i")};
        expRowIndices = new int[]{0, 0, 2, 2, 3, 4, 4, 5, 5, 5, 9, 10, 11, 11, 12, 12, 13, 14, 14, 15, 16, 17, 18, 19, 23, 23, 24, 24, 27, 27, 27};
        expColIndices = new int[]{5, 8, 6, 9, 8, 7, 10, 7, 9, 10, 10, 8, 6, 9, 7, 8, 10, 1, 4, 4, 1, 0, 1, 2, 3, 4, 2, 4, 1, 3, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));
    }


    @Test
    void realDenseInvDirectSumTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        double[][] bEntries;
        MatrixOld b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new CNumber[]{new CNumber("0.70966+0.15842i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.80057, 0.28729, 0.5102},
                {0.85163, 0.92955, 0.79641},
                {0.77585, 0.35028, 0.9436},
                {0.23878, 0.59249, 0.27167}};
        b = new MatrixOld(bEntries);

        expShape = new Shape(6, 6);
        expEntries = new CNumber[]{new CNumber("0.80057"), new CNumber("0.28729"), new CNumber("0.5102"), new CNumber("0.85163"), new CNumber("0.92955"), new CNumber("0.79641"), new CNumber("0.77585"), new CNumber("0.35028"), new CNumber("0.9436"), new CNumber("0.23878"), new CNumber("0.59249"), new CNumber("0.27167"), new CNumber("0.70966+0.15842i")};
        expRowIndices = new int[]{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
        expColIndices = new int[]{3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.16785, 0.51452},
                {0.55687, 0.00919},
                {0.91283, 0.71861},
                {0.4706, 0.63073},
                {0.94686, 0.93044}};
        b = new MatrixOld(bEntries);

        expShape = new Shape(6, 4);
        expEntries = new CNumber[]{new CNumber("0.16785"), new CNumber("0.51452"), new CNumber("0.55687"), new CNumber("0.00919"), new CNumber("0.91283"), new CNumber("0.71861"), new CNumber("0.4706"), new CNumber("0.63073"), new CNumber("0.94686"), new CNumber("0.93044")};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
        expColIndices = new int[]{2, 3, 2, 3, 2, 3, 2, 3, 2, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new CNumber[]{new CNumber("0.47183+0.10623i"), new CNumber("0.61911+0.36706i"), new CNumber("0.67903+0.7145i"), new CNumber("0.07196+0.84351i"), new CNumber("0.33506+0.85722i"), new CNumber("0.09925+0.02582i"), new CNumber("0.91219+0.08732i"), new CNumber("0.40247+0.03899i"), new CNumber("0.59776+0.27702i"), new CNumber("0.67777+0.95102i"), new CNumber("0.16611+0.84191i"), new CNumber("0.13517+0.59676i"), new CNumber("0.2599+0.34769i"), new CNumber("0.20792+0.93642i")};
        aRowIndices = new int[]{0, 1, 2, 2, 5, 6, 7, 7, 7, 8, 8, 9, 12, 13};
        aColIndices = new int[]{2, 3, 1, 2, 3, 1, 0, 2, 3, 0, 3, 1, 4, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.54728, 0.81285, 0.89945, 0.67697, 0.68489, 0.93281},
                {0.52002, 0.6991, 0.3612, 0.23775, 0.30479, 0.69227},
                {0.70478, 0.96067, 0.61074, 0.50879, 0.36487, 0.25989},
                {0.7988, 0.96244, 0.25853, 0.37059, 0.52823, 0.75686},
                {0.4544, 0.5076, 0.64476, 0.39994, 0.78403, 0.76487},
                {0.95245, 0.76454, 0.90096, 0.93676, 0.58304, 0.4057},
                {0.25202, 0.79335, 0.44649, 0.1608, 0.06315, 0.92318},
                {0.57744, 0.55908, 0.93852, 0.27638, 0.76867, 0.24151},
                {0.42211, 0.01012, 0.10379, 0.26216, 0.955, 0.38766},
                {0.77631, 0.7091, 0.47902, 0.32255, 0.92555, 0.38567},
                {0.14936, 0.55688, 0.80156, 0.65858, 0.5423, 0.42977},
                {0.21615, 0.31412, 0.0202, 0.1485, 0.47591, 0.77931},
                {0.43203, 0.63658, 0.50725, 0.14705, 0.58838, 0.57095},
                {0.39932, 0.35901, 0.56806, 0.97636, 0.10387, 0.51516}};
        b = new MatrixOld(bEntries);

        expShape = new Shape(28, 11);
        expEntries = new CNumber[]{new CNumber("0.54728"), new CNumber("0.81285"), new CNumber("0.89945"), new CNumber("0.67697"), new CNumber("0.68489"), new CNumber("0.93281"), new CNumber("0.52002"), new CNumber("0.6991"), new CNumber("0.3612"), new CNumber("0.23775"), new CNumber("0.30479"), new CNumber("0.69227"), new CNumber("0.70478"), new CNumber("0.96067"), new CNumber("0.61074"), new CNumber("0.50879"), new CNumber("0.36487"), new CNumber("0.25989"), new CNumber("0.7988"), new CNumber("0.96244"), new CNumber("0.25853"), new CNumber("0.37059"), new CNumber("0.52823"), new CNumber("0.75686"), new CNumber("0.4544"), new CNumber("0.5076"), new CNumber("0.64476"), new CNumber("0.39994"), new CNumber("0.78403"), new CNumber("0.76487"), new CNumber("0.95245"), new CNumber("0.76454"), new CNumber("0.90096"), new CNumber("0.93676"), new CNumber("0.58304"), new CNumber("0.4057"), new CNumber("0.25202"), new CNumber("0.79335"), new CNumber("0.44649"), new CNumber("0.1608"), new CNumber("0.06315"), new CNumber("0.92318"), new CNumber("0.57744"), new CNumber("0.55908"), new CNumber("0.93852"), new CNumber("0.27638"), new CNumber("0.76867"), new CNumber("0.24151"), new CNumber("0.42211"), new CNumber("0.01012"), new CNumber("0.10379"), new CNumber("0.26216"), new CNumber("0.955"), new CNumber("0.38766"), new CNumber("0.77631"), new CNumber("0.7091"), new CNumber("0.47902"), new CNumber("0.32255"), new CNumber("0.92555"), new CNumber("0.38567"), new CNumber("0.14936"), new CNumber("0.55688"), new CNumber("0.80156"), new CNumber("0.65858"), new CNumber("0.5423"), new CNumber("0.42977"), new CNumber("0.21615"), new CNumber("0.31412"), new CNumber("0.0202"), new CNumber("0.1485"), new CNumber("0.47591"), new CNumber("0.77931"), new CNumber("0.43203"), new CNumber("0.63658"), new CNumber("0.50725"), new CNumber("0.14705"), new CNumber("0.58838"), new CNumber("0.57095"), new CNumber("0.39932"), new CNumber("0.35901"), new CNumber("0.56806"), new CNumber("0.97636"), new CNumber("0.10387"), new CNumber("0.51516"), new CNumber("0.47183+0.10623i"), new CNumber("0.61911+0.36706i"), new CNumber("0.67903+0.7145i"), new CNumber("0.07196+0.84351i"), new CNumber("0.33506+0.85722i"), new CNumber("0.09925+0.02582i"), new CNumber("0.91219+0.08732i"), new CNumber("0.40247+0.03899i"), new CNumber("0.59776+0.27702i"), new CNumber("0.67777+0.95102i"), new CNumber("0.16611+0.84191i"), new CNumber("0.13517+0.59676i"), new CNumber("0.2599+0.34769i"), new CNumber("0.20792+0.93642i")};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 15, 16, 16, 19, 20, 21, 21, 21, 22, 22, 23, 26, 27};
        expColIndices = new int[]{5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 2, 3, 1, 2, 3, 1, 0, 2, 3, 0, 3, 1, 4, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));
    }


    @Test
    void complexDenseInvDirectSumTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        CNumber[][] bEntries;
        CMatrixOld b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new CNumber[]{new CNumber("0.01371+0.45798i")};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.653+0.44012i"), new CNumber("0.96493+0.74667i"), new CNumber("0.92319+0.8181i")},
                {new CNumber("0.12956+0.89688i"), new CNumber("0.36787+0.52372i"), new CNumber("0.4675+0.93904i")},
                {new CNumber("0.96875+0.65718i"), new CNumber("0.83108+0.73656i"), new CNumber("0.49703+0.6304i")},
                {new CNumber("0.90662+0.08429i"), new CNumber("0.55422+0.87839i"), new CNumber("0.18309+0.49541i")}};
        b = new CMatrixOld(bEntries);

        expShape = new Shape(6, 6);
        expEntries = new CNumber[]{new CNumber("0.653+0.44012i"), new CNumber("0.96493+0.74667i"), new CNumber("0.92319+0.8181i"), new CNumber("0.12956+0.89688i"), new CNumber("0.36787+0.52372i"), new CNumber("0.4675+0.93904i"), new CNumber("0.96875+0.65718i"), new CNumber("0.83108+0.73656i"), new CNumber("0.49703+0.6304i"), new CNumber("0.90662+0.08429i"), new CNumber("0.55422+0.87839i"), new CNumber("0.18309+0.49541i"), new CNumber("0.01371+0.45798i")};
        expRowIndices = new int[]{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 5};
        expColIndices = new int[]{3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.74831+0.7366i"), new CNumber("0.31032+0.33495i")},
                {new CNumber("0.42562+0.80464i"), new CNumber("0.89781+0.33133i")},
                {new CNumber("0.29559+0.64901i"), new CNumber("0.12285+0.43343i")},
                {new CNumber("0.04944+0.1112i"), new CNumber("0.33989+0.81211i")},
                {new CNumber("0.936+0.22087i"), new CNumber("0.66966+0.54744i")}};
        b = new CMatrixOld(bEntries);

        expShape = new Shape(6, 4);
        expEntries = new CNumber[]{new CNumber("0.74831+0.7366i"), new CNumber("0.31032+0.33495i"), new CNumber("0.42562+0.80464i"), new CNumber("0.89781+0.33133i"), new CNumber("0.29559+0.64901i"), new CNumber("0.12285+0.43343i"), new CNumber("0.04944+0.1112i"), new CNumber("0.33989+0.81211i"), new CNumber("0.936+0.22087i"), new CNumber("0.66966+0.54744i")};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
        expColIndices = new int[]{2, 3, 2, 3, 2, 3, 2, 3, 2, 3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new CNumber[]{new CNumber("0.46962+0.17868i"), new CNumber("0.33693+0.93647i"), new CNumber("0.50236+0.59781i"), new CNumber("0.73101+0.07755i"), new CNumber("0.96895+0.00506i"), new CNumber("0.37699+0.07088i"), new CNumber("0.72397+0.49475i"), new CNumber("0.98482+0.0973i"), new CNumber("0.18821+0.13868i"), new CNumber("0.47866+0.10774i"), new CNumber("0.03012+0.57197i"), new CNumber("0.73759+0.60487i"), new CNumber("0.1654+0.27707i"), new CNumber("0.99406+0.83751i")};
        aRowIndices = new int[]{1, 2, 4, 4, 6, 8, 8, 9, 10, 10, 11, 11, 12, 13};
        aColIndices = new int[]{0, 1, 0, 3, 1, 2, 4, 3, 1, 4, 1, 2, 3, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.51513+0.07676i"), new CNumber("0.85445+0.81327i"), new CNumber("0.63273+0.65983i"), new CNumber("0.07243+0.28635i"), new CNumber("0.53554+0.60701i"), new CNumber("0.77908+0.88646i")},
                {new CNumber("0.26678+0.27533i"), new CNumber("0.02189+0.35006i"), new CNumber("0.83419+0.53774i"), new CNumber("0.21729+0.91947i"), new CNumber("0.98107+0.33525i"), new CNumber("0.33811+0.7859i")},
                {new CNumber("0.21147+0.39761i"), new CNumber("0.27744+0.77517i"), new CNumber("0.89765+0.0772i"), new CNumber("0.8978+0.77165i"), new CNumber("0.02569+0.06173i"), new CNumber("0.59092+0.90457i")},
                {new CNumber("0.26189+0.90656i"), new CNumber("0.39334+0.80734i"), new CNumber("0.13819+0.07125i"), new CNumber("0.39565+0.13149i"), new CNumber("0.34629+0.56198i"), new CNumber("0.80122+0.57958i")},
                {new CNumber("0.41252+0.79495i"), new CNumber("0.72799+0.68814i"), new CNumber("0.95731+0.38063i"), new CNumber("0.34746+0.51576i"), new CNumber("0.16116+0.36335i"), new CNumber("0.19533+0.87831i")},
                {new CNumber("0.9556+0.49289i"), new CNumber("0.36637+0.47281i"), new CNumber("0.5024+0.86343i"), new CNumber("0.10118+0.10255i"), new CNumber("0.45618+0.9362i"), new CNumber("0.33215+0.42905i")},
                {new CNumber("0.0492+0.15099i"), new CNumber("0.68106+0.40926i"), new CNumber("0.90839+0.37491i"), new CNumber("0.88676+0.29334i"), new CNumber("0.25131+0.81344i"), new CNumber("0.59329+0.7539i")},
                {new CNumber("0.66093+0.83296i"), new CNumber("0.24694+0.7055i"), new CNumber("0.00301+0.482i"), new CNumber("0.4863+0.78445i"), new CNumber("0.66447+0.3601i"), new CNumber("0.63681+0.63447i")},
                {new CNumber("0.71261+0.61154i"), new CNumber("0.158+0.70848i"), new CNumber("0.37956+0.46261i"), new CNumber("0.50126+0.50071i"), new CNumber("0.82288+0.33121i"), new CNumber("0.91691+0.55077i")},
                {new CNumber("0.89142+0.10647i"), new CNumber("0.20105+0.50321i"), new CNumber("0.24035+0.86039i"), new CNumber("0.13335+0.00151i"), new CNumber("0.64584+0.8288i"), new CNumber("0.99423+0.96267i")},
                {new CNumber("0.41475+0.52903i"), new CNumber("0.29702+0.55817i"), new CNumber("0.75457+0.48185i"), new CNumber("0.67387+0.29825i"), new CNumber("0.06314+0.47099i"), new CNumber("0.74414+0.98149i")},
                {new CNumber("0.84972+0.31614i"), new CNumber("0.35738+0.78219i"), new CNumber("0.95908+0.5187i"), new CNumber("0.13973+0.22577i"), new CNumber("0.53332+0.58431i"), new CNumber("0.95873+0.8393i")},
                {new CNumber("0.56784+0.54242i"), new CNumber("0.61272+0.75044i"), new CNumber("0.28474+0.65693i"), new CNumber("0.69311+0.14116i"), new CNumber("0.8687+0.00501i"), new CNumber("0.04146+0.81794i")},
                {new CNumber("0.30756+0.90481i"), new CNumber("0.56671+0.95885i"), new CNumber("0.61453+0.03104i"), new CNumber("0.914+0.05092i"), new CNumber("0.90644+0.25179i"), new CNumber("0.49604+0.07538i")}};
        b = new CMatrixOld(bEntries);

        expShape = new Shape(28, 11);
        expEntries = new CNumber[]{new CNumber("0.51513+0.07676i"), new CNumber("0.85445+0.81327i"), new CNumber("0.63273+0.65983i"), new CNumber("0.07243+0.28635i"), new CNumber("0.53554+0.60701i"), new CNumber("0.77908+0.88646i"), new CNumber("0.26678+0.27533i"), new CNumber("0.02189+0.35006i"), new CNumber("0.83419+0.53774i"), new CNumber("0.21729+0.91947i"), new CNumber("0.98107+0.33525i"), new CNumber("0.33811+0.7859i"), new CNumber("0.21147+0.39761i"), new CNumber("0.27744+0.77517i"), new CNumber("0.89765+0.0772i"), new CNumber("0.8978+0.77165i"), new CNumber("0.02569+0.06173i"), new CNumber("0.59092+0.90457i"), new CNumber("0.26189+0.90656i"), new CNumber("0.39334+0.80734i"), new CNumber("0.13819+0.07125i"), new CNumber("0.39565+0.13149i"), new CNumber("0.34629+0.56198i"), new CNumber("0.80122+0.57958i"), new CNumber("0.41252+0.79495i"), new CNumber("0.72799+0.68814i"), new CNumber("0.95731+0.38063i"), new CNumber("0.34746+0.51576i"), new CNumber("0.16116+0.36335i"), new CNumber("0.19533+0.87831i"), new CNumber("0.9556+0.49289i"), new CNumber("0.36637+0.47281i"), new CNumber("0.5024+0.86343i"), new CNumber("0.10118+0.10255i"), new CNumber("0.45618+0.9362i"), new CNumber("0.33215+0.42905i"), new CNumber("0.0492+0.15099i"), new CNumber("0.68106+0.40926i"), new CNumber("0.90839+0.37491i"), new CNumber("0.88676+0.29334i"), new CNumber("0.25131+0.81344i"), new CNumber("0.59329+0.7539i"), new CNumber("0.66093+0.83296i"), new CNumber("0.24694+0.7055i"), new CNumber("0.00301+0.482i"), new CNumber("0.4863+0.78445i"), new CNumber("0.66447+0.3601i"), new CNumber("0.63681+0.63447i"), new CNumber("0.71261+0.61154i"), new CNumber("0.158+0.70848i"), new CNumber("0.37956+0.46261i"), new CNumber("0.50126+0.50071i"), new CNumber("0.82288+0.33121i"), new CNumber("0.91691+0.55077i"), new CNumber("0.89142+0.10647i"), new CNumber("0.20105+0.50321i"), new CNumber("0.24035+0.86039i"), new CNumber("0.13335+0.00151i"), new CNumber("0.64584+0.8288i"), new CNumber("0.99423+0.96267i"), new CNumber("0.41475+0.52903i"), new CNumber("0.29702+0.55817i"), new CNumber("0.75457+0.48185i"), new CNumber("0.67387+0.29825i"), new CNumber("0.06314+0.47099i"), new CNumber("0.74414+0.98149i"), new CNumber("0.84972+0.31614i"), new CNumber("0.35738+0.78219i"), new CNumber("0.95908+0.5187i"), new CNumber("0.13973+0.22577i"), new CNumber("0.53332+0.58431i"), new CNumber("0.95873+0.8393i"), new CNumber("0.56784+0.54242i"), new CNumber("0.61272+0.75044i"), new CNumber("0.28474+0.65693i"), new CNumber("0.69311+0.14116i"), new CNumber("0.8687+0.00501i"), new CNumber("0.04146+0.81794i"), new CNumber("0.30756+0.90481i"), new CNumber("0.56671+0.95885i"), new CNumber("0.61453+0.03104i"), new CNumber("0.914+0.05092i"), new CNumber("0.90644+0.25179i"), new CNumber("0.49604+0.07538i"), new CNumber("0.46962+0.17868i"), new CNumber("0.33693+0.93647i"), new CNumber("0.50236+0.59781i"), new CNumber("0.73101+0.07755i"), new CNumber("0.96895+0.00506i"), new CNumber("0.37699+0.07088i"), new CNumber("0.72397+0.49475i"), new CNumber("0.98482+0.0973i"), new CNumber("0.18821+0.13868i"), new CNumber("0.47866+0.10774i"), new CNumber("0.03012+0.57197i"), new CNumber("0.73759+0.60487i"), new CNumber("0.1654+0.27707i"), new CNumber("0.99406+0.83751i")};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 15, 16, 18, 18, 20, 22, 22, 23, 24, 24, 25, 25, 26, 27};
        expColIndices = new int[]{5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 0, 1, 0, 3, 1, 2, 4, 3, 1, 4, 1, 2, 3, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, DirectSum.invDirectSum(a, b));
    }
}
