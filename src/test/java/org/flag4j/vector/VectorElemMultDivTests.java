package org.flag4j.vector;

import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class VectorElemMultDivTests {

    int[] indices;

    double[] aEntries;
    VectorOld a;

    @Test
    void realDenseMultTestCase() {
        double[] bEntries, expEntries;
        VectorOld b, exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        bEntries = new double[]{8.55, -9.133, -8.34};
        b = new VectorOld(bEntries);
        expEntries = new double[]{aEntries[0]*bEntries[0], aEntries[1]*bEntries[1], aEntries[2]*bEntries[2]};
        exp = new VectorOld(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        bEntries = new double[]{8.55, -9.133, -8.34, 23.4};
        b = new VectorOld(bEntries);

        VectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void complexDenseMultTestCase() {
        CNumber[] bEntries, expEntries;
        CVectorOld b, exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        bEntries = new CNumber[]{new CNumber(23.456, -9234), new CNumber(0, 8.234), new CNumber(-9234.5, 0.24)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[]{bEntries[0].mult(aEntries[0]),bEntries[1].mult(aEntries[1]), bEntries[2].mult(aEntries[2])};
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.elemMult(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        bEntries = new CNumber[]{new CNumber(23.456, -9234), new CNumber(0, 8.234)};
        b = new CVectorOld(bEntries);

        CVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void realSparseMultTestCase() {
        int[] expIndices;
        double[] bEntries, expEntries;
        CooVectorOld b, exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        bEntries = new double[]{8.55};
        indices = new int[]{2};
        b = new CooVectorOld(3, bEntries, indices);
        expEntries = new double[]{aEntries[2]*bEntries[0]};
        expIndices = new int[]{2};
        exp = new CooVectorOld(3, expEntries, expIndices);

        CooVectorOld act = a.elemMult(b);

        assertEquals(exp.size, act.size);
        assertArrayEquals(exp.entries, act.entries);
        assertArrayEquals(exp.indices, act.indices);

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        bEntries = new double[]{8.55};
        indices = new int[]{2};
        b = new CooVectorOld(402, bEntries, indices);

        CooVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }


    @Test
    void complexSparseMultTestCase() {
        int[] expIndices;
        CNumber[] bEntries, expEntries;
        CooCVectorOld b, exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        bEntries = new CNumber[]{new CNumber(-9.234, 1.5)};
        indices = new int[]{2};
        b = new CooCVectorOld(3, bEntries, indices);
        expEntries = new CNumber[]{bEntries[0].mult(aEntries[2])};
        expIndices = new int[]{2};
        exp = new CooCVectorOld(3, expEntries, expIndices);

        CooCVectorOld act = a.elemMult(b);

        assertEquals(exp.size, act.size);
        assertArrayEquals(exp.entries, act.entries);
        assertArrayEquals(exp.indices, act.indices);

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        bEntries = new CNumber[]{new CNumber(-9.234, 1.5)};
        indices = new int[]{2};
        b = new CooCVectorOld(31, bEntries, indices);

        CooCVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemMult(finalB));
    }

    // ---------------------------------------------------------------------------------------------------------------
    @Test
    void realDenseDivTestCase() {
        double[] bEntries, expEntries;
        VectorOld b, exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        bEntries = new double[]{8.55, -9.133, -8.34};
        b = new VectorOld(bEntries);
        expEntries = new double[]{aEntries[0]/bEntries[0], aEntries[1]/bEntries[1], aEntries[2]/bEntries[2]};
        exp = new VectorOld(expEntries);

        assertEquals(exp, a.elemDiv(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        bEntries = new double[]{8.55, -9.133, -8.34, 23.4};
        b = new VectorOld(bEntries);

        VectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemDiv(finalB));
    }


    @Test
    void complexDenseDivTestCase() {
        CNumber[] bEntries, expEntries;
        CVectorOld b, exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        bEntries = new CNumber[]{new CNumber(23.456, -9234), new CNumber(0, 8.234), new CNumber(-9234.5, 0.24)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[]{new CNumber(aEntries[0]).div(bEntries[0]),
                new CNumber(aEntries[1]).div(bEntries[1]), new CNumber(aEntries[2]).div(bEntries[2])};
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.elemDiv(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        bEntries = new CNumber[]{new CNumber(23.456, -9234), new CNumber(0, 8.234)};
        b = new CVectorOld(bEntries);

        CVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.elemDiv(finalB));
    }
}
