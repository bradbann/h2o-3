package hex;

import water.Iced;
import water.Key;
import water.exceptions.H2OIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;
import water.util.VecUtils;

import java.util.Arrays;

public class FoldAssignment extends Iced<FoldAssignment> {
    final Vec _fold;
    final int[] _foldMapping;
    final boolean _internal;

    FoldAssignment(Vec fold, int[] foldMapping, boolean internal) {
        _fold = fold;
        _foldMapping = inverseMapping(foldMapping);
        _internal = internal;
    }

    Frame toFrame(Key<Frame> key) {
        return new Frame(key, new String[]{"fold_assignment"}, new Vec[]{_fold});
    }
    
    void dispose() {
        if (_internal)
            _fold.remove();
        Arrays.fill(_foldMapping, -1);
    }
    
    static FoldAssignment fromUserFoldSpecification(int N, Vec fold) {
        if( !fold.isInt() ||
                (!(fold.min() == 0 && fold.max() == N-1) &&
                        !(fold.min() == 1 && fold.max() == N))) // Allow 0 to N-1, or 1 to N
            throw new H2OIllegalArgumentException("Fold column must be either categorical or contiguous integers from 0..N-1 or 1..N");

        return new FoldAssignment(fold, foldMapping(fold), false);
    }

    static int nFoldWork(Vec fold) {
        return foldMapping(fold).length;
    }
    
    static FoldAssignment fromInternalFold(int N, Vec fold) {
        return new FoldAssignment(fold, ArrayUtils.seq(0, N), true);
    }

    static int[] inverseMapping(int[] foldMapping) {
        int max = -1;
        for (int foldIdx : foldMapping) {
            if (foldIdx > max)
                max = foldIdx;
        }
        final int[] idxToFoldNumber = new int[max + 1];
        Arrays.fill(idxToFoldNumber, -1);
        for (int i = 0; i < foldMapping.length; i++) {
            idxToFoldNumber[foldMapping[i]] = i;
        }
        return idxToFoldNumber;
    }

    static int[] foldMapping(Vec f) {
        Vec fc = VecUtils.toCategoricalVec(f);
        final String[] actualDomain;
        try {
            if (!f.isCategorical()) {
                actualDomain = fc.domain();
            } else {
                actualDomain = VecUtils.collectDomainFast(fc);
            }
        } finally {
            fc.remove();
        }
        int N = actualDomain.length;
        if (Arrays.equals(actualDomain, fc.domain())) {
            int offset = f.isCategorical() ? 0 : (int) f.min();
            return ArrayUtils.seq(offset, N + offset);
        } else {
            int[] mapping = new int[N];
            String[] fullDomain = fc.domain();
            for (int i = 0; i < N; i++) {
                int pos = ArrayUtils.find(fullDomain, actualDomain[i]);
                assert pos >= 0;
                mapping[i] = pos;
            }
            return mapping;
        }
    }

}
