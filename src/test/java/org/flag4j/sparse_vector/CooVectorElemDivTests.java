package org.flag4j.sparse_vector;

import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooVectorElemDivTests {

    CooVectorOld a;
    int size;

    @Test
    void denseElemDivTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 4};
        size = 6;
        a = new CooVectorOld(size, aValues, aIndices);

        double[] bValues;
        int[] expIndices;
        VectorOld b;
        double[] expValues;
        CooVectorOld exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51};
        b = new VectorOld(bValues);

        expValues = new double[]{1.34/1.223, 51.6/3.4, -0.00245/14.5};
        expIndices = new int[]{0, 2, 4};
        exp = new CooVectorOld(size, expValues, expIndices);
        assertEquals(exp, a.elemDiv(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new double[]{1.223, -44.51, 3.4, 2.3, 14.5, -14.51, 123, 25.2, 155};
        b = new VectorOld(bValues);

        VectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemDiv(finalB));
    }


    @Test
    void denseComplexElemDivTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245};
        int[] aIndices = {0, 2, 4};
        size = 6;
        a = new CooVectorOld(size, aValues, aIndices);

        CNumber[] bValues, expValues;
        int[] expIndices;
        CVectorOld b;
        CooCVectorOld exp;

        // -------------------- Sub-case 1 --------------------
        bValues = new CNumber[]{new CNumber(24.3, -0.013), new CNumber(0, 13.6),
                new CNumber(2.4), new CNumber(-994.1 ,1.45), new CNumber(1495, 13.4),
                new CNumber(9924.515, 51.5)};
        b = new CVectorOld(bValues);

        expValues = new CNumber[]{
                new CNumber(1.34).div(new CNumber(24.3, -0.013)),
                new CNumber(51.6).div(new CNumber(2.4)),
                new CNumber(-0.00245).div(new CNumber(1495, 13.4))
        };
        expIndices = new int[]{0, 2, 4};
        exp = new CooCVectorOld(size, expValues, expIndices);
        assertEquals(exp, a.elemDiv(b));

        // -------------------- Sub-case 2 --------------------
        bValues = new CNumber[]{new CNumber(24.3, -0.013), new CNumber(0, 13.6)};
        b = new CVectorOld(bValues);

        CVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemDiv(finalB));
    }


    @Test
    void doubleScalarElemDivTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245, 99.2456, -1005.6};
        int[] aIndices = {2, 5, 81, 102, 104};
        size = 151;
        a = new CooVectorOld(size, aValues, aIndices);

        double b;

        double[] expValues;
        int[] expIndices;
        CooVectorOld exp;

        // -------------------- Sub-case 1 --------------------
        b = 24.56;

        expValues = new double[]{1.34/b, 51.6/b, -0.00245/b, 99.2456/b, -1005.6/b};
        expIndices = new int[]{2, 5, 81, 102, 104};
        exp = new CooVectorOld(151, expValues, expIndices);
        assertEquals(exp, a.div(b));
    }


    @Test
    void complexScalarElemDivTestCase() {
        double[] aValues = {1.34, 51.6, -0.00245, 99.2456, -1005.6};
        int[] aIndices = {2, 5, 81, 102, 104};
        size = 151;
        a = new CooVectorOld(size, aValues, aIndices);

        CNumber b;

        CNumber[] expValues;
        int[] expIndices;
        CooCVectorOld exp;

        // -------------------- Sub-case 1 --------------------
        b = new CNumber(234.6677, -9.35);

        expValues = new CNumber[]{
                new CNumber(1.34).div(b), new CNumber(51.6).div(b),
                new CNumber(-0.00245).div(b), new CNumber(99.2456).div(b),
                new CNumber(-1005.6).div(b)};
        expIndices = new int[]{2, 5, 81, 102, 104};
        exp = new CooCVectorOld(151, expValues, expIndices);

        assertEquals(exp, a.div(b));
    }
}
