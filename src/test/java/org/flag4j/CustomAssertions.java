package org.flag4j;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;

import static org.junit.jupiter.api.Assertions.*;

public final class CustomAssertions {


    /**
     * Checks if matrices are equal counting NaN equal to NaN.
     * @param exp Expected MatrixOld.
     * @param act Actual MatrixOld.
     */
    public static void assertEqualsNaN(MatrixOld exp, MatrixOld act) {
        assertEquals(exp.shape, act.shape);
        for(int i=0; i<exp.entries.length; i++) {
            if(Double.isNaN(exp.entries[i])) {
                assertTrue(Double.isNaN(act.entries[i]));
            } else {
                assertEquals(exp.entries[i], act.entries[i], 0);
            }
        }
    }


    /**
     * Checks if matrices are equal counting NaN equal to NaN.
     * @param exp Expected MatrixOld.
     * @param act Actual MatrixOld.
     */
    public static void assertEqualsNaN(CMatrixOld exp, CMatrixOld act) {
        assertEquals(exp.shape, act.shape);
        for(int i=0; i<exp.entries.length; i++) {
            if(exp.entries[i].isNaN()) {
                if(Double.isNaN(exp.entries[i].re)) {
                    assertTrue(Double.isNaN(act.entries[i].re));
                } else {
                    assertEquals(exp.entries[i].re, act.entries[i].re, 0);
                }

                if(Double.isNaN(exp.entries[i].im)) {
                    assertTrue(Double.isNaN(act.entries[i].im));
                } else {
                    assertEquals(exp.entries[i].im, act.entries[i].im, 0);
                }
            } else {
                assertEquals(exp.entries[i], act.entries[i]);
            }
        }
    }


    /**
     * Checks if sparse matrices are equal counting NaN equal to NaN.
     * @param exp Expected MatrixOld.
     * @param act Actual MatrixOld.
     */
    public static void assertEqualsNaN(CooMatrixOld exp, CooMatrixOld act) {
        assertEquals(exp.shape, act.shape);
        assertArrayEquals(exp.rowIndices, act.colIndices);
        for(int i=0; i<exp.entries.length; i++) {
            if(Double.isNaN(exp.entries[i])) {
                assertTrue(Double.isNaN(act.entries[i]));
            } else {
                assertEquals(exp.entries[i], act.entries[i], 0);
            }
        }
    }


    /**
     * Checks if sparse vectors are equal counting NaN equal to NaN.
     * @param exp Expected vector.
     * @param act Actual vector.
     */
    public static void assertEqualsNaN(CooVectorOld exp, CooVectorOld act) {
        assertEquals(exp.shape, act.shape);
        assertArrayEquals(exp.indices, act.indices);
        for(int i=0; i<exp.entries.length; i++) {
            if(Double.isNaN(exp.entries[i])) {
                assertTrue(Double.isNaN(act.entries[i]));
            } else {
                assertEquals(exp.entries[i], act.entries[i], 0);
            }
        }
    }
}
