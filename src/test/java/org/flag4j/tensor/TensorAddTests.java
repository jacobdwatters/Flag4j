package org.flag4j.tensor;

import org.flag4j.arrays_old.dense.CTensorOld;
import org.flag4j.arrays_old.dense.TensorOld;
import org.flag4j.arrays_old.sparse.CooCTensorOld;
import org.flag4j.arrays_old.sparse.CooTensorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorAddTests {
    static double[] aEntries;
    static TensorOld A;
    static Shape aShape, bShape, expShape;

    int[][] sparseIndices;

    @BeforeEach
    void setup() {
        aEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        aShape = new Shape(2, 3, 2);
        A = new TensorOld(aShape, aEntries);
    }

    @Test
    void realDenseTestCase() {
        double[] bEntries, expEntries;
        TensorOld B, exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new TensorOld(bShape, bEntries);
        expEntries = new double[]{
            aEntries[0]+bEntries[0], aEntries[1]+bEntries[1], aEntries[2]+bEntries[2],
            aEntries[3]+bEntries[3], aEntries[4]+bEntries[4], aEntries[5]+bEntries[5],
            aEntries[6]+bEntries[6], aEntries[7]+bEntries[7], aEntries[8]+bEntries[8],
            aEntries[9]+bEntries[9], aEntries[10]+bEntries[10], aEntries[11]+bEntries[11]
        };
        expShape = new Shape(2, 3, 2);
        exp = new TensorOld(expShape, expEntries);

        assertEquals(exp, A.add(B));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new TensorOld(bShape, bEntries);

        TensorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.add(finalB));

        // ----------------------- Sub-case 3 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0, 1.4, 5
        };
        bShape = new Shape(7, 2);
        B = new TensorOld(bShape, bEntries);

        TensorOld finalB1 = B;
        assertThrows(LinearAlgebraException.class, ()->A.add(finalB1));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries, expEntries;
        CooTensorOld B;
        TensorOld exp;

        // ------------------------- Sub-case 1 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape( 2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new CooTensorOld(bShape, bEntries, sparseIndices);
        expEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        expShape = new Shape(2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])] += bEntries[0];
        expEntries[expShape.entriesIndex(sparseIndices[1])] += bEntries[1];
        expEntries[expShape.entriesIndex(sparseIndices[2])] += bEntries[2];
        exp = new TensorOld(expShape, expEntries);

        assertEquals(exp, A.add(B));

        // ------------------------- Sub-case 2 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(4, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new CooTensorOld(bShape, bEntries, sparseIndices);

        CooTensorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.add(finalB));
    }


    @Test
    void complexDenseTestCase() {
        CNumber[] bEntries, expEntries;
        CTensorOld B, exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new CNumber[]{
                new CNumber(-0.00234, 2.452), new CNumber(15.6), new CNumber(99.2442, 9.1), 
                new CNumber(100.252, 1235), new CNumber(-78.2556, -99.1441), new CNumber(0.111134, -772.4),
                new CNumber(671.455, 15.56), new CNumber(-0.00024), new CNumber(515.667, 895.52), 
                new CNumber(14.515), new CNumber(100.135), new CNumber(0, 1)
        };
        bShape = new Shape(2, 3, 2);
        B = new CTensorOld(bShape, bEntries);
        expEntries = new CNumber[]{
                bEntries[0].add(aEntries[0]), bEntries[1].add(aEntries[1]), bEntries[2].add(aEntries[2]),
                bEntries[3].add(aEntries[3]), bEntries[4].add(aEntries[4]), bEntries[5].add(aEntries[5]),
                bEntries[6].add(aEntries[6]), bEntries[7].add(aEntries[7]), bEntries[8].add(aEntries[8]),
                bEntries[9].add(aEntries[9]), bEntries[10].add(aEntries[10]), bEntries[11].add(aEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensorOld(expShape, expEntries);

        assertEquals(exp, A.add(B));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new CNumber[]{
                new CNumber(-0.00234, 2.452), new CNumber(15.6), new CNumber(99.2442, 9.1),
                new CNumber(100.252, 1235), new CNumber(-78.2556, -99.1441), new CNumber(0.111134, -772.4),
                new CNumber(671.455, 15.56), new CNumber(-0.00024), new CNumber(515.667, 895.52),
                new CNumber(14.515), new CNumber(100.135), new CNumber(0, 1)
        };
        bShape = new Shape(12);
        B = new CTensorOld(bShape, bEntries);

        CTensorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.add(finalB));
    }


    @Test
    void complexSparseTestCase() {
        CNumber[] bEntries, expEntries;
        CooCTensorOld B;
        CTensorOld exp;

        // ------------------------- Sub-case 1 -------------------------
        bEntries = new CNumber[]{
                new CNumber(1, -0.2045), new CNumber(-800.145, 3204.5)
        };
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}
        };
        B = new CooCTensorOld(bShape, bEntries, sparseIndices);
        expEntries = new CNumber[]{
                new CNumber(1.23), new CNumber(2.556), new CNumber(-121.5), new CNumber(15.61), new CNumber(14.15), new CNumber(-99.23425),
                new CNumber(0.001345), new CNumber(2.677), new CNumber(8.14), new CNumber(-0.000194), new CNumber(1), new CNumber(234)
        };
        expShape = new Shape(2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])] = expEntries[expShape.entriesIndex(sparseIndices[0])].add(bEntries[0]);
        expEntries[expShape.entriesIndex(sparseIndices[1])] = expEntries[expShape.entriesIndex(sparseIndices[1])].add(bEntries[1]);
        exp = new CTensorOld(expShape, expEntries);

        assertEquals(exp, A.add(B));

        // ------------------------- Sub-case 2 -------------------------
        bEntries = new CNumber[]{
                new CNumber(1, -0.2045), new CNumber(-800.145, 3204.5)
        };
        bShape = new Shape(13, 89, 14576);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}
        };
        B = new CooCTensorOld(bShape, bEntries, sparseIndices);

        CooCTensorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.add(finalB));
    }


    @Test
    void addRealScalarTestCase() {
        double[] expEntries;
        TensorOld exp;
        double b;

        // ----------------------- Sub-case 1 -----------------------
        b = 234.25256;
        expEntries = new double[]{
                aEntries[0]+b, aEntries[1]+b, aEntries[2]+b,
                aEntries[3]+b, aEntries[4]+b, aEntries[5]+b,
                aEntries[6]+b, aEntries[7]+b, aEntries[8]+b,
                aEntries[9]+b, aEntries[10]+b, aEntries[11]+b
        };
        expShape = new Shape(2, 3, 2);
        exp = new TensorOld(expShape, expEntries);

        assertEquals(exp, A.add(b));
    }


    @Test
    void addComplexScalar() {
        CNumber[] expEntries;
        CTensorOld exp;
        CNumber b;

        // ----------------------- Sub-case 1 -----------------------
        b = new CNumber(234.5, -364.00);
        expEntries = new CNumber[]{
                b.add(aEntries[0]), b.add(aEntries[1]), b.add(aEntries[2]),
                b.add(aEntries[3]), b.add(aEntries[4]), b.add(aEntries[5]),
                b.add(aEntries[6]), b.add(aEntries[7]), b.add(aEntries[8]),
                b.add(aEntries[9]), b.add(aEntries[10]), b.add(aEntries[11])
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensorOld(expShape, expEntries);

        assertEquals(exp, A.add(b));
    }


    @Test
    void realDenseAddEqTestCase() {
        double[] bEntries, expEntries;
        TensorOld B, exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2);
        B = new TensorOld(bShape, bEntries);
        expEntries = new double[]{
                aEntries[0]+bEntries[0], aEntries[1]+bEntries[1], aEntries[2]+bEntries[2],
                aEntries[3]+bEntries[3], aEntries[4]+bEntries[4], aEntries[5]+bEntries[5],
                aEntries[6]+bEntries[6], aEntries[7]+bEntries[7], aEntries[8]+bEntries[8],
                aEntries[9]+bEntries[9], aEntries[10]+bEntries[10], aEntries[11]+bEntries[11]
        };
        expShape = new Shape(2, 3, 2);
        exp = new TensorOld(expShape, expEntries);

        A.addEq(B);
        assertEquals(exp, A);

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0
        };
        bShape = new Shape(2, 3, 2, 1);
        B = new TensorOld(bShape, bEntries);

        TensorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.addEq(finalB));

        // ----------------------- Sub-case 3 -----------------------
        bEntries = new double[]{
                -0.00234, 15.6, 99.2442, 100.252, -78.2556, 0.111134,
                671.455, -0.00024, 515.667, 14.515, 100.135, 0, 1.4, 5
        };
        bShape = new Shape(7, 2);
        B = new TensorOld(bShape, bEntries);

        TensorOld finalB1 = B;
        assertThrows(LinearAlgebraException.class, ()->A.addEq(finalB1));
    }


    @Test
    void realSparseAddEqTestCase() {
        double[] bEntries, expEntries;
        CooTensorOld B;
        TensorOld exp;

        // ------------------------- Sub-case 1 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(2, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new CooTensorOld(bShape, bEntries, sparseIndices);
        expEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        expShape = new Shape(2, 3, 2);
        expEntries[expShape.entriesIndex(sparseIndices[0])] += bEntries[0];
        expEntries[expShape.entriesIndex(sparseIndices[1])] += bEntries[1];
        expEntries[expShape.entriesIndex(sparseIndices[2])] += bEntries[2];
        exp = new TensorOld(expShape, expEntries);

        A.addEq(B);
        assertEquals(exp, A);

        // ------------------------- Sub-case 2 -------------------------
        bEntries = new double[]{
                1.34, -0.0245, 8001.1
        };
        bShape = new Shape(4, 3, 2);
        sparseIndices = new int[][]{
                {0, 2, 1}, {1, 1, 0}, {1, 2, 1}
        };
        B = new CooTensorOld(bShape, bEntries, sparseIndices);

        CooTensorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.addEq(finalB));
    }


    @Test
    void addEqRealScalarTestCase() {
        double[] expEntries;
        TensorOld exp;
        double b;

        // ----------------------- Sub-case 1 -----------------------
        b = 234.25256;
        expEntries = new double[]{
                aEntries[0]+b, aEntries[1]+b, aEntries[2]+b,
                aEntries[3]+b, aEntries[4]+b, aEntries[5]+b,
                aEntries[6]+b, aEntries[7]+b, aEntries[8]+b,
                aEntries[9]+b, aEntries[10]+b, aEntries[11]+b
        };
        expShape = new Shape(2, 3, 2);
        exp = new TensorOld(expShape, expEntries);

        A.addEq(b);
        assertEquals(exp, A);
    }
}
