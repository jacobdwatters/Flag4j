package org.flag4j.tensor;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.dense.CTensor;
import org.flag4j.dense.Tensor;
import org.flag4j.sparse.CooCTensor;
import org.flag4j.sparse.CooTensor;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorElemMultTests {
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
                aEntries[0]*bEntries[0], aEntries[1]*bEntries[1], aEntries[2]*bEntries[2],
                aEntries[3]*bEntries[3], aEntries[4]*bEntries[4], aEntries[5]*bEntries[5],
                aEntries[6]*bEntries[6], aEntries[7]*bEntries[7], aEntries[8]*bEntries[8],
                aEntries[9]*bEntries[9], aEntries[10]*bEntries[10], aEntries[11]*bEntries[11]
        };
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.elemMult(B));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new Tensor(bShape, bEntries);

        Tensor finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB));

        // ----------------------- Sub-case 3 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0, 1.4, 5
        };
        bShape = new Shape(7, 2);
        B = new Tensor(bShape, bEntries);

        Tensor finalB1 = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB1));
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
        bShape = new Shape(true, 2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);
        expEntries = new double[aEntries.length];
        expShape = new Shape(true,2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])] = bEntries[0]*aEntries[expShape.entriesIndex(sparseIndices[0])];
        expEntries[expShape.entriesIndex(sparseIndices[1])] = bEntries[1]*aEntries[expShape.entriesIndex(sparseIndices[1])];
        expEntries[expShape.entriesIndex(sparseIndices[2])] = bEntries[2]*aEntries[expShape.entriesIndex(sparseIndices[2])];
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.elemMult(B));

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
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB));
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
                bEntries[0].mult(aEntries[0]), bEntries[1].mult(aEntries[1]), bEntries[2].mult(aEntries[2]),
                bEntries[3].mult(aEntries[3]), bEntries[4].mult(aEntries[4]), bEntries[5].mult(aEntries[5]),
                bEntries[6].mult(aEntries[6]), bEntries[7].mult(aEntries[7]), bEntries[8].mult(aEntries[8]),
                bEntries[9].mult(aEntries[9]), bEntries[10].mult(aEntries[10]), bEntries[11].mult(aEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.elemMult(B));

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
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB));
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
        bShape = new Shape(true, 2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);
        expEntries = new CNumber[]{
                new CNumber(), new CNumber(), new CNumber(), new CNumber(), new CNumber(), new CNumber(),
                new CNumber(), new CNumber(), new CNumber(), new CNumber(), new CNumber(), new CNumber()
        };
        expShape = new Shape(true, 2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])] = bEntries[0].mult(aEntries[expShape.entriesIndex(sparseIndices[0])]);
        expEntries[expShape.entriesIndex(sparseIndices[1])] = bEntries[1].mult(aEntries[expShape.entriesIndex(sparseIndices[1])]);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.elemMult(B));

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
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB));
    }
}
