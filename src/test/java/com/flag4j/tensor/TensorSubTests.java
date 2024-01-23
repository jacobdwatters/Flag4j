package com.flag4j.tensor;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorSubTests {
    static double[] aEntries;
    static Tensor A;
    static Shape aShape, bShape, expShape;

    int[][] sparseIndices;

    @BeforeEach
    void setup() {
        aEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        aShape = new Shape(2, 3, 2);
        A = new Tensor(aShape, aEntries);
    }

    @Test
    void realDenseTestCase() {
        double[] bEntries, expEntries;
        Tensor B, exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new Tensor(bShape, bEntries);
        expEntries = new double[]{
                aEntries[0]-bEntries[0], aEntries[1]-bEntries[1], aEntries[2]-bEntries[2],
                aEntries[3]-bEntries[3], aEntries[4]-bEntries[4], aEntries[5]-bEntries[5],
                aEntries[6]-bEntries[6], aEntries[7]-bEntries[7], aEntries[8]-bEntries[8],
                aEntries[9]-bEntries[9], aEntries[10]-bEntries[10], aEntries[11]-bEntries[11]
        };
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(expShape, expEntries);

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
        double[] bEntries, expEntries;
        CooTensor B;
        Tensor exp;

        // ------------------------- Sub-case 1 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(true,2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);
        expEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        expShape = new Shape(true,2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])] -= bEntries[0];
        expEntries[expShape.entriesIndex(sparseIndices[1])] -= bEntries[1];
        expEntries[expShape.entriesIndex(sparseIndices[2])] -= bEntries[2];
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.sub(B));

        // ------------------------- Sub-case 2 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(true,4, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);

        CooTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.sub(finalB));
    }


    @Test
    void complexDenseTestCase() {
        CNumber[] bEntries, expEntries;
        CTensor B, exp;

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
                new CNumber(aEntries[0]).sub(bEntries[0]), new CNumber(aEntries[1]).sub(bEntries[1]), new CNumber(aEntries[2]).sub(bEntries[2]),
                new CNumber(aEntries[3]).sub(bEntries[3]), new CNumber(aEntries[4]).sub(bEntries[4]), new CNumber(aEntries[5]).sub(bEntries[5]),
                new CNumber(aEntries[6]).sub(bEntries[6]), new CNumber(aEntries[7]).sub(bEntries[7]), new CNumber(aEntries[8]).sub(bEntries[8]),
                new CNumber(aEntries[9]).sub(bEntries[9]), new CNumber(aEntries[10]).sub(bEntries[10]), new CNumber(aEntries[11]).sub(bEntries[11])
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
        CNumber[] bEntries, expEntries;
        CooCTensor B;
        CTensor exp;

        // ------------------------- Sub-case 1 -------------------------
        bEntries = new CNumber[]{
                new CNumber(1, -0.2045), new CNumber(-800.145, 3204.5)
        };
        bShape = new Shape(true,2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);
        expEntries = new CNumber[]{
                new CNumber(1.23), new CNumber(2.556), new CNumber(-121.5), new CNumber(15.61), new CNumber(14.15), new CNumber(-99.23425),
                new CNumber(0.001345), new CNumber(2.677), new CNumber(8.14), new CNumber(-0.000194), new CNumber(1), new CNumber(234)
        };
        expShape = new Shape(true,2, 3, 2);
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
        B = new CooCTensor(bShape, bEntries, sparseIndices);

        CooCTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.sub(finalB));
    }


    @Test
    void subRealScalar() {
        double[] expEntries;
        Tensor exp;
        double b;

        // ----------------------- Sub-case 1 -----------------------
        b = 234.25256;
        expEntries = new double[]{
                aEntries[0]-b, aEntries[1]-b, aEntries[2]-b,
                aEntries[3]-b, aEntries[4]-b, aEntries[5]-b,
                aEntries[6]-b, aEntries[7]-b, aEntries[8]-b,
                aEntries[9]-b, aEntries[10]-b, aEntries[11]-b
        };
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(expShape, expEntries);

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
                new CNumber(aEntries[0]).sub(b), new CNumber(aEntries[1]).sub(b), new CNumber(aEntries[2]).sub(b),
                new CNumber(aEntries[3]).sub(b), new CNumber(aEntries[4]).sub(b), new CNumber(aEntries[5]).sub(b),
                new CNumber(aEntries[6]).sub(b), new CNumber(aEntries[7]).sub(b), new CNumber(aEntries[8]).sub(b),
                new CNumber(aEntries[9]).sub(b), new CNumber(aEntries[10]).sub(b), new CNumber(aEntries[11]).sub(b)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.sub(b));
    }


    @Test
    void realDenseSubEqTestCase() {
        double[] bEntries, expEntries;
        Tensor B, exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new Tensor(bShape, bEntries);
        expEntries = new double[]{
                aEntries[0]-bEntries[0], aEntries[1]-bEntries[1], aEntries[2]-bEntries[2],
                aEntries[3]-bEntries[3], aEntries[4]-bEntries[4], aEntries[5]-bEntries[5],
                aEntries[6]-bEntries[6], aEntries[7]-bEntries[7], aEntries[8]-bEntries[8],
                aEntries[9]-bEntries[9], aEntries[10]-bEntries[10], aEntries[11]-bEntries[11]
        };
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(expShape, expEntries);

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
        double[] bEntries, expEntries;
        CooTensor B;
        Tensor exp;

        // ------------------------- Sub-case 1 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(true,2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);
        expEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        expShape = new Shape(true,2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])] -= bEntries[0];
        expEntries[expShape.entriesIndex(sparseIndices[1])] -= bEntries[1];
        expEntries[expShape.entriesIndex(sparseIndices[2])] -= bEntries[2];
        exp = new Tensor(expShape, expEntries);

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
        B = new CooTensor(bShape, bEntries, sparseIndices);

        CooTensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.subEq(finalB));
    }


    @Test
    void subEqRealScalarTestCase() {
        double[] expEntries;
        Tensor exp;
        double b;

        // ----------------------- Sub-case 1 -----------------------
        b = 234.25256;
        expEntries = new double[]{
                aEntries[0]-b, aEntries[1]-b, aEntries[2]-b,
                aEntries[3]-b, aEntries[4]-b, aEntries[5]-b,
                aEntries[6]-b, aEntries[7]-b, aEntries[8]-b,
                aEntries[9]-b, aEntries[10]-b, aEntries[11]-b
        };
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(expShape, expEntries);

        A.subEq(b);
        assertEquals(exp, A);
    }
}
