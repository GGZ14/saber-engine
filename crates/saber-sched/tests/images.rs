//! D3: image requests through the scheduler. The vision tower runs ONCE per request, at the head
//! of the queue and before the group is admitted; an executor with no tower refuses the request at
//! submit, by name; a tower failure ends the request with its reason (never a prefill over rows
//! nobody computed).

mod common;

use std::sync::{Arc, Mutex};

use common::{drain, request, scheduler};
use saber_sched::{
    AdmittedSeq, MockExecutor, SeqId, SeqScript, StepBatch, StepExecutor, StepResult,
};
use saber_serve::engine::{Engine, EngineError, FinishReason, ImageBatch, RequestId};

/// What the wrapped executor saw, in order.
type Log = Arc<Mutex<Vec<String>>>;

#[derive(Debug)]
struct VisionMock {
    inner: MockExecutor,
    log: Log,
    fail: bool,
}

impl StepExecutor for VisionMock {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        let who: Vec<String> = batch
            .slots
            .iter()
            .map(|s| s.request_id.to_string())
            .collect();
        self.log
            .lock()
            .unwrap()
            .push(format!("step {:?} {}", batch.phase, who.join(",")));
        self.inner.execute(batch)
    }
    fn admit(&mut self, seq: &AdmittedSeq) {
        self.log
            .lock()
            .unwrap()
            .push(format!("admit {}", seq.request_id));
        self.inner.admit(seq);
    }
    fn release(&mut self, seq: SeqId) {
        self.inner.release(seq);
    }
    fn serves_images(&self) -> bool {
        true
    }
    fn encode_images(&mut self, request: RequestId, images: &ImageBatch) -> Result<(), String> {
        self.log.lock().unwrap().push(format!(
            "tower {request} {} images {} patches",
            images.grids.len(),
            images.patches()
        ));
        if self.fail {
            Err("vision: the tower failed (injected)".to_owned())
        } else {
            Ok(())
        }
    }
}

fn batch() -> ImageBatch {
    ImageBatch {
        grids: vec![[1, 16, 16], [1, 24, 32]],
        pixel_values: vec![0.0; (256 + 768) * 1536],
    }
}

#[tokio::test]
async fn a_text_only_executor_refuses_an_image_request_by_name() {
    let sched = scheduler(MockExecutor::new());
    match sched.submit_with_images(request(&[1, 2, 3], 4), batch()) {
        Err(EngineError::Unsupported { param, message }) => {
            assert_eq!(param, "image");
            assert!(message.contains("TEXT ONLY"), "{message}");
        }
        Err(other) => panic!("wrong refusal: {other}"),
        Ok(_) => panic!("a text-only executor accepted an image request"),
    }
}

#[tokio::test]
async fn the_tower_runs_once_before_admission() {
    let log: Log = Arc::default();
    let mut inner = MockExecutor::new();
    inner.push_script(SeqScript::per_token([7, 8]));
    let sched = scheduler(VisionMock {
        inner,
        log: Arc::clone(&log),
        fail: false,
    });
    let sub = sched
        .submit_with_images(request(&[1, 2, 3], 2), batch())
        .unwrap();
    for _ in 0..8 {
        sched.step();
    }
    let _ = drain(sub).await;
    let log: Vec<String> = log
        .lock()
        .unwrap()
        .iter()
        .filter(|l| !l.starts_with("step "))
        .cloned()
        .collect();
    assert_eq!(log.len(), 2, "{log:?}");
    assert!(
        log[0].starts_with("tower ") && log[0].ends_with("2 images 1024 patches"),
        "{log:?}"
    );
    assert!(log[1].starts_with("admit "), "{log:?}");
}

#[tokio::test]
async fn a_tower_failure_ends_the_request_with_its_reason() {
    let log: Log = Arc::default();
    let sched = scheduler(VisionMock {
        inner: MockExecutor::new(),
        log: Arc::clone(&log),
        fail: true,
    });
    let sub = sched
        .submit_with_images(request(&[1, 2, 3], 2), batch())
        .unwrap();
    sched.step();
    let events = drain(sub).await;
    let last = events.last().expect("an event");
    assert_eq!(last.finished, Some(FinishReason::Abort));
    assert!(last.error.as_deref().unwrap().contains("injected"));
    assert!(
        log.lock().unwrap().iter().all(|l| !l.starts_with("admit")),
        "never admitted"
    );
}

/// D4's slot law: the tower has ONE output slot, so a second image request is not encoded while
/// the first one's prefill has not consumed its rows -- and it is encoded as soon as it has.
#[tokio::test]
async fn a_second_image_request_waits_for_the_first_ones_prefill() {
    let log: Log = Arc::default();
    let mut inner = MockExecutor::new();
    inner.push_script(SeqScript::per_token([7, 8, 9]));
    inner.push_script(SeqScript::per_token([7, 8, 9]));
    let sched = scheduler(VisionMock {
        inner,
        log: Arc::clone(&log),
        fail: false,
    });
    let a = sched
        .submit_with_images(request(&[1, 2, 3], 3), batch())
        .unwrap();
    let b = sched
        .submit_with_images(request(&[4, 5, 6], 3), batch())
        .unwrap();
    for _ in 0..16 {
        sched.step();
    }
    let _ = drain(a).await;
    let _ = drain(b).await;
    let log = log.lock().unwrap().clone();
    let towers: Vec<usize> = log
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("tower "))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(towers.len(), 2, "{log:?}");
    let first = log[towers[0]].split(' ').nth(1).unwrap().to_owned();
    let first_prefill = log
        .iter()
        .position(|l| {
            l.starts_with("step ") && l.split(' ').nth(2).unwrap().split(',').any(|x| x == first)
        })
        .unwrap_or_else(|| panic!("the first request prefilled: {log:?}"));
    assert!(
        first_prefill < towers[1],
        "the second tower ran before the first request's first step consumed its rows: {log:?}"
    );
    let waits = sched.counters().vision_waits;
    assert!(waits >= 1, "the wait is counted: {waits}");
}
