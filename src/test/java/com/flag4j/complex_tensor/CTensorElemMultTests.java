package com.flag4j.complex_tensor;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.Shape;
import com.flag4j.dense.CTensor;
import com.flag4j.dense.Tensor;
import com.flag4j.sparse.CooCTensor;
import com.flag4j.sparse.CooTensor;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTensorElemMultTests {
    static CNumber[] aEntries ,expEntries;
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
                aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]), aEntries[2].mult(bEntries[2]),
                aEntries[3].mult(bEntries[3]), aEntries[4].mult(bEntries[4]), aEntries[5].mult(bEntries[5]),
                aEntries[6].mult(bEntries[6]), aEntries[7].mult(bEntries[7]), aEntries[8].mult(bEntries[8]),
                aEntries[9].mult(bEntries[9]), aEntries[10].mult(bEntries[10]), aEntries[11].mult(bEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

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
        double[] bEntries;
        CooTensor B;

        // ------------------------- Sub-case 1 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(true, 2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new CooTensor(bShape, bEntries, sparseIndices);
        expEntries = new CNumber[aEntries.length];
        ArrayUtils.fillZeros(expEntries);
        expShape = new Shape(true, 2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])] = aEntries[expShape.entriesIndex(sparseIndices[0])].mult(bEntries[0]);
        expEntries[expShape.entriesIndex(sparseIndices[1])] = aEntries[expShape.entriesIndex(sparseIndices[1])].mult(bEntries[1]);
        expEntries[expShape.entriesIndex(sparseIndices[2])] = aEntries[expShape.entriesIndex(sparseIndices[2])].mult(bEntries[2]);
        exp = new CTensor(expShape, expEntries);

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
                aEntries[0].mult(bEntries[0]), aEntries[1].mult(bEntries[1]), aEntries[2].mult(bEntries[2]),
                aEntries[3].mult(bEntries[3]), aEntries[4].mult(bEntries[4]), aEntries[5].mult(bEntries[5]),
                aEntries[6].mult(bEntries[6]), aEntries[7].mult(bEntries[7]), aEntries[8].mult(bEntries[8]),
                aEntries[9].mult(bEntries[9]), aEntries[10].mult(bEntries[10]), aEntries[11].mult(bEntries[11])
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
        CNumber[] bEntries;
        CooCTensor B;

        // ------------------------- Sub-case 1 -------------------------
        bEntries = new CNumber[]{
                new CNumber(1, -0.2045), new CNumber(-800.145, 3204.5)
        };
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}
        };
        B = new CooCTensor(bShape, bEntries, sparseIndices);
        expEntries = new CNumber[]{
                new CNumber(), new CNumber(), new CNumber(), new CNumber(), new CNumber(), new CNumber(),
                new CNumber(), new CNumber(), new CNumber(), new CNumber(), new CNumber(), new CNumber()
        };
        expShape = new Shape(true, 2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])] = aEntries[expShape.entriesIndex(sparseIndices[0])].mult(bEntries[0]);
        expEntries[expShape.entriesIndex(sparseIndices[1])] = aEntries[expShape.entriesIndex(sparseIndices[1])].mult(bEntries[1]);
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
