package com.flag4j.complex_tensor;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.exceptions.LinearAlgebraException;
import com.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTensorSubTests {
    static CNumber[] aEntries, expEntries;
    static CTensor A, exp;
    static Shape aShape, bShape, expShape;

    int[][] sparseIndices;

    @BeforeEach
    void setup() {
        aEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(235.61, 1.45), new CNumber(0, -0.00024),
                new CNumber(1.0), new CNumber(-85.1, 9.234), new CNumber(1.345, -781.2),
                new CNumber(0.014, -2.45),  new CNumber(-140.0),  new CNumber(0, 1.5),
                new CNumber(51.0, 24.56),  new CNumber(6.1, -0.03),  new CNumber(-0.00014, 1.34),};
        aShape = new Shape(2, 3, 2);
        A = new CTensor(aShape, aEntries);
    }


    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Tensor B;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new Tensor(bShape, bEntries);
        expEntries = new CNumber[]{
                aEntries[0].sub(bEntries[0]), aEntries[1].sub(bEntries[1]), aEntries[2].sub(bEntries[2]),
                aEntries[3].sub(bEntries[3]), aEntries[4].sub(bEntries[4]), aEntries[5].sub(bEntries[5]),
                aEntries[6].sub(bEntries[6]), aEntries[7].sub(bEntries[7]), aEntries[8].sub(bEntries[8]),
                aEntries[9].sub(bEntries[9]), aEntries[10].sub(bEntries[10]), aEntries[11].sub(bEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(B));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new Tensor(bShape, bEntries);

        Tensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.sub(finalB));

        // ----------------------- Sub-case 3 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0, 1.4, 5
        };
        bShape = new Shape(7, 2);
        B = new Tensor(bShape, bEntries);

        Tensor finalB1 = B;
        assertThrows(LinearAlgebraException.class, ()->A.sub(finalB1));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        SparseTensor B;

        // ------------------------- Sub-case 1 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new SparseTensor(bShape, bEntries, sparseIndices);
        expEntries = ArrayUtils.copyOf(aEntries);
        expShape = new Shape(2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])].subEq(bEntries[0]);
        expEntries[expShape.entriesIndex(sparseIndices[1])].subEq(bEntries[1]);
        expEntries[expShape.entriesIndex(sparseIndices[2])].subEq(bEntries[2]);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(B));

        // ------------------------- Sub-case 2 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(4, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new SparseTensor(bShape, bEntries, sparseIndices);

        SparseTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.sub(finalB));
    }


    @Test
    void complexDenseTestCase() {
        CNumber[] bEntries;
        CTensor B;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new CNumber[]{
                new CNumber(-0.00234, 2.452), new CNumber(15.6), new CNumber(99.2442, 9.1),
                new CNumber(100.252, 1235), new CNumber(-78.2556, -99.1441), new CNumber(0.111134, -772.4),
                new CNumber(671.455, 15.56), new CNumber(-0.00024), new CNumber(515.667, 895.52),
                new CNumber(14.515), new CNumber(100.135), new CNumber(0, 1)
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensor(bShape, bEntries);
        expEntries = new CNumber[]{
                aEntries[0].sub(bEntries[0]), aEntries[1].sub(bEntries[1]), aEntries[2].sub(bEntries[2]),
                aEntries[3].sub(bEntries[3]), aEntries[4].sub(bEntries[4]), aEntries[5].sub(bEntries[5]),
                aEntries[6].sub(bEntries[6]), aEntries[7].sub(bEntries[7]), aEntries[8].sub(bEntries[8]),
                aEntries[9].sub(bEntries[9]), aEntries[10].sub(bEntries[10]), aEntries[11].sub(bEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(B));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new CNumber[]{
                new CNumber(-0.00234, 2.452), new CNumber(15.6), new CNumber(99.2442, 9.1),
                new CNumber(100.252, 1235), new CNumber(-78.2556, -99.1441), new CNumber(0.111134, -772.4),
                new CNumber(671.455, 15.56), new CNumber(-0.00024), new CNumber(515.667, 895.52),
                new CNumber(14.515), new CNumber(100.135), new CNumber(0, 1)
        };
        bShape = new Shape(12);
        B = new CTensor(bShape, bEntries);

        CTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.sub(finalB));
    }


    @Test
    void complexSparseTestCase() {
        CNumber[] bEntries;
        SparseCTensor B;

        // ------------------------- Sub-case 1 -------------------------
        bEntries = new CNumber[]{
                new CNumber(1, -0.2045), new CNumber(-800.145, 3204.5)
        };
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}
        };
        B = new SparseCTensor(bShape, bEntries, sparseIndices);
        expEntries = ArrayUtils.copyOf(aEntries);
        expShape = new Shape(2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])].subEq(bEntries[0]);
        expEntries[expShape.entriesIndex(sparseIndices[1])].subEq(bEntries[1]);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(B));

        // ------------------------- Sub-case 2 -------------------------
        bEntries = new CNumber[]{
                new CNumber(1, -0.2045), new CNumber(-800.145, 3204.5)
        };
        bShape = new Shape(13, 89, 14576);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}
        };
        B = new SparseCTensor(bShape, bEntries, sparseIndices);

        SparseCTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.sub(finalB));
    }


    @Test
    void subRealScalarTestCase() {
        double b;

        // ----------------------- Sub-case 1 -----------------------
        b = 234.25256;
        expEntries = new CNumber[]{
                aEntries[0].sub(b), aEntries[1].sub(b), aEntries[2].sub(b),
                aEntries[3].sub(b), aEntries[4].sub(b), aEntries[5].sub(b),
                aEntries[6].sub(b), aEntries[7].sub(b), aEntries[8].sub(b),
                aEntries[9].sub(b), aEntries[10].sub(b), aEntries[11].sub(b)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(b));
    }


    @Test
    void subComplexScalar() {
        CNumber[] expEntries;
        CTensor exp;
        CNumber b;

        // ----------------------- Sub-case 1 -----------------------
        b = new CNumber(234.5, -364.00);
        expEntries = new CNumber[]{
                aEntries[0].sub(b), aEntries[1].sub(b), aEntries[2].sub(b),
                aEntries[3].sub(b), aEntries[4].sub(b), aEntries[5].sub(b),
                aEntries[6].sub(b), aEntries[7].sub(b), aEntries[8].sub(b),
                aEntries[9].sub(b), aEntries[10].sub(b), aEntries[11].sub(b)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(b));
    }


    @Test
    void realDenseSubEqTestCase() {
        double[] bEntries;
        Tensor B;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new Tensor(bShape, bEntries);
        expEntries = new CNumber[]{
                aEntries[0].sub(bEntries[0]), aEntries[1].sub(bEntries[1]), aEntries[2].sub(bEntries[2]),
                aEntries[3].sub(bEntries[3]), aEntries[4].sub(bEntries[4]), aEntries[5].sub(bEntries[5]),
                aEntries[6].sub(bEntries[6]), aEntries[7].sub(bEntries[7]), aEntries[8].sub(bEntries[8]),
                aEntries[9].sub(bEntries[9]), aEntries[10].sub(bEntries[10]), aEntries[11].sub(bEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        A.subEq(B);
        assertEquals(exp, A);

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new Tensor(bShape, bEntries);

        Tensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB));

        // ----------------------- Sub-case 3 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0, 1.4, 5
        };
        bShape = new Shape(7, 2);
        B = new Tensor(bShape, bEntries);

        Tensor finalB1 = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB1));
    }


    @Test
    void realSparseSubEqTestCase() {
        double[] bEntries;
        SparseTensor B;

        // ------------------------- Sub-case 1 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new SparseTensor(bShape, bEntries, sparseIndices);
        expEntries = ArrayUtils.copyOf(aEntries);
        expShape = new Shape(2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])].subEq(bEntries[0]);
        expEntries[expShape.entriesIndex(sparseIndices[1])].subEq(bEntries[1]);
        expEntries[expShape.entriesIndex(sparseIndices[2])].subEq(bEntries[2]);
        exp = new CTensor(expShape, expEntries);

        A.subEq(B);
        assertEquals(exp, A);

        // ------------------------- Sub-case 2 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(4, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new SparseTensor(bShape, bEntries, sparseIndices);

        SparseTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB));
    }


    @Test
    void subEqRealScalarTestCase() {
        double b;

        // ----------------------- Sub-case 1 -----------------------
        b = 234.25256;
        expEntries = new CNumber[]{
                aEntries[0].sub(b), aEntries[1].sub(b), aEntries[2].sub(b),
                aEntries[3].sub(b), aEntries[4].sub(b), aEntries[5].sub(b),
                aEntries[6].sub(b), aEntries[7].sub(b), aEntries[8].sub(b),
                aEntries[9].sub(b), aEntries[10].sub(b), aEntries[11].sub(b)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        A.subEq(b);
        assertEquals(exp, A);
    }


    @Test
    void complexDenseSubEqTestCase() {
        CNumber[] bEntries;
        CTensor B;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new CNumber[]{
                new CNumber(1.34, -0.324), new CNumber(0.134), new CNumber(0, 2.501),
                new CNumber(-994.1, 0.0234), new CNumber(9.4, 0.14), new CNumber(5.2, 1104.5),
                new CNumber(103.45, 6), new CNumber(-23.45, 1.4), new CNumber(-2, -0.4),
                new CNumber(3.55), new CNumber(), new CNumber(100.2456),
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensor(bShape, bEntries);
        expEntries = new CNumber[]{
                aEntries[0].sub(bEntries[0]), aEntries[1].sub(bEntries[1]), aEntries[2].sub(bEntries[2]),
                aEntries[3].sub(bEntries[3]), aEntries[4].sub(bEntries[4]), aEntries[5].sub(bEntries[5]),
                aEntries[6].sub(bEntries[6]), aEntries[7].sub(bEntries[7]), aEntries[8].sub(bEntries[8]),
                aEntries[9].sub(bEntries[9]), aEntries[10].sub(bEntries[10]), aEntries[11].sub(bEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        A.subEq(B);
        assertEquals(exp, A);

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new CNumber[]{
                new CNumber(1.34, -0.324), new CNumber(0.134), new CNumber(0, 2.501),
                new CNumber(-994.1, 0.0234), new CNumber(9.4, 0.14), new CNumber(5.2, 1104.5),
                new CNumber(103.45, 6), new CNumber(-23.45, 1.4), new CNumber(-2, -0.4),
                new CNumber(3.55), new CNumber(), new CNumber(100.2456),
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new CTensor(bShape, bEntries);

        CTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB));

        // ----------------------- Sub-case 3 -----------------------
        bEntries = new CNumber[]{
                new CNumber(1.34, -0.324), new CNumber(0.134), new CNumber(0, 2.501),
                new CNumber(-994.1, 0.0234), new CNumber(9.4, 0.14), new CNumber(5.2, 1104.5),
                new CNumber(103.45, 6), new CNumber(-23.45, 1.4), new CNumber(-2, -0.4),
                new CNumber(3.55), new CNumber(), new CNumber(100.2456),
                new CNumber(1.344), new CNumber(0.924, 55.6)
        };
        bShape = new Shape(7, 2);
        B = new CTensor(bShape, bEntries);

        CTensor finalB1 = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB1));
    }


    @Test
    void complexSparseSubEqTestCase() {
        CNumber[] bEntries;
        SparseCTensor B;

        // ------------------------- Sub-case 1 -------------------------
        bEntries = new CNumber[]{
                new CNumber(13, 0.244), new CNumber(9.345), new CNumber(0, -9.124)
        };
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new SparseCTensor(bShape, bEntries, sparseIndices);
        expEntries = ArrayUtils.copyOf(aEntries);
        expShape = new Shape(2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])].subEq(bEntries[0]);
        expEntries[expShape.entriesIndex(sparseIndices[1])].subEq(bEntries[1]);
        expEntries[expShape.entriesIndex(sparseIndices[2])].subEq(bEntries[2]);
        exp = new CTensor(expShape, expEntries);

        A.subEq(B);
        assertEquals(exp, A);

        // ------------------------- Sub-case 2 -------------------------
        bEntries = new CNumber[]{
                new CNumber(13, 0.244), new CNumber(9.345), new CNumber(0, -9.124)
        };
        bShape = new Shape(4, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new SparseCTensor(bShape, bEntries, sparseIndices);

        SparseCTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB));
    }


    @Test
    void subEqComplexScalarTestCase() {
        CNumber b;

        // ----------------------- Sub-case 1 -----------------------
        b = new CNumber(234.25256, -1451.13455);
        expEntries = new CNumber[]{
                aEntries[0].sub(b), aEntries[1].sub(b), aEntries[2].sub(b),
                aEntries[3].sub(b), aEntries[4].sub(b), aEntries[5].sub(b),
                aEntries[6].sub(b), aEntries[7].sub(b), aEntries[8].sub(b),
                aEntries[9].sub(b), aEntries[10].sub(b), aEntries[11].sub(b)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        A.subEq(b);
        assertEquals(exp, A);
    }
}
